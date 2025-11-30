import os
import time
import random
import string
import threading
import json
import logging
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, Any, Optional, List

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.helpers import escape_markdown

# -------------------- Configuration --------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PLAYERS_FILE = DATA_DIR / "players.json"
CODES_FILE = DATA_DIR / "codes.json"
PENDING_FILE = DATA_DIR / "pending.json"
LOG_FILE = "bot.log"

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable required. Set BOT_TOKEN before running.")

ADMIN_ENV = os.getenv("ADMIN_ID")
if not ADMIN_ENV:
    raise RuntimeError("ADMIN_ID environment variable is required. Set ADMIN_ID to your Telegram user id.")
try:
    ADMIN_ID = int(ADMIN_ENV)
except ValueError:
    raise RuntimeError("ADMIN_ID environment variable must be an integer (your Telegram user id).")

ROUND_PLAYER_COUNT = 10
COUNTDOWN_SECONDS = 5
TAP_DURATION_SECONDS = 10
TIEBREAK_WINDOW = 5
PENDING_PAYOUT_SECONDS = 180

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# -------------------- Optional keep-alive HTTP server --------------------
class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is alive!")

def run_server():
    port = int(os.getenv("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), SimpleHandler)
    logger.info("HTTP keep-alive server starting on port %s", port)
    server.serve_forever()

threading.Thread(target=run_server, daemon=True).start()

# -------------------- Persistence helpers --------------------
def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load %s; returning empty dict", path)
        return {}

def _save_json_atomic(path: Path, data: Any):
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except Exception:
        logger.exception("Failed to save %s", path)

# -------------------- App initialization --------------------
app = ApplicationBuilder().token(BOT_TOKEN).build()

# -------------------- State (persisted) --------------------
players: Dict[str, Dict[str, Any]] = _load_json(PLAYERS_FILE)
active_codes: Dict[str, Dict[str, Any]] = _load_json(CODES_FILE)
pending_payout: Optional[Dict[str, Any]] = _load_json(PENDING_FILE) or None

# -------------------- Runtime state --------------------
current_round_queue: List[int] = []
round_running: bool = False
round_lock = asyncio.Lock()
round_start_time: Optional[float] = None
round_message_map: Dict[int, int] = {}

# -------------------- Utilities --------------------
def make_username_display(pdata: Dict[str, Any]) -> str:
    if not pdata:
        return "Unknown"
    if pdata.get("username"):
        return f"@{pdata['username']}"
    return pdata.get("first_name") or f"User{pdata.get('user_id','?')}"

def gen_code(length: int = 8) -> str:
    return "TTW" + "".join(random.choices(string.ascii_uppercase + string.digits, k=length))

def persist_players():
    _save_json_atomic(PLAYERS_FILE, players)
    logger.info("Saved players (%d)", len(players))

def persist_codes():
    _save_json_atomic(CODES_FILE, active_codes)
    logger.info("Saved codes (%d)", len(active_codes))

def persist_pending():
    global pending_payout
    if pending_payout:
        _save_json_atomic(PENDING_FILE, pending_payout)
        logger.info("Saved pending payout")
    else:
        try:
            if PENDING_FILE.exists():
                PENDING_FILE.unlink()
                logger.info("Removed pending payout file")
        except Exception:
            logger.exception("Failed to remove pending file")

async def safe_send(chat_id: int, text: str, **kwargs) -> Optional[Message]:
    try:
        return await app.bot.send_message(chat_id=chat_id, text=text, **kwargs)
    except Exception:
        logger.exception("Failed to send message to %s", chat_id)
        return None

# -------------------- Player commands --------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 *Welcome to Tap-to-Win!* \n\n"
        "⚠️ *Disclaimer:* Only stake money you can afford to lose.\n\n"
        "To play:\n"
        "1️⃣ Register: `/register 07XXXXXXXX`\n"
        "2️⃣ Wait for admin to generate a one-time code\n"
        "3️⃣ Use `/entergame` then send the code when prompted\n\n"
        "Type `/help` for commands. Good luck! 🍀"
    )
    await update.message.reply_markdown_v2(text)

async def cmd_register(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("Usage: /register 07XXXXXXXX (e.g., /register 0712345678) 📱")
        return
    phone = context.args[0].strip()
    user = update.effective_user
    uid_str = str(user.id)
    players[uid_str] = {
        "user_id": user.id,
        "username": user.username,
        "first_name": user.first_name,
        "phone": phone,
        "chat_id": update.effective_chat.id,
        "taps": 0,
        "taps_first5": 0,
    }
    persist_players()
    await update.message.reply_text(f"✅ Registered {make_username_display(players[uid_str])} — {phone} 👍")

async def cmd_entergame(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    uid_str = str(user.id)
    if uid_str not in players:
        await update.message.reply_text("❌ You must register first with /register.")
        return
    context.user_data["awaiting_code"] = True
    await update.message.reply_text("🔐 Send your ONE-TIME game code now (as a normal message).")

async def handler_text_codes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("awaiting_code"):
        return
    code = update.message.text.strip().upper()
    user = update.effective_user
    uid_int = user.id
    uid_str = str(uid_int)
    if code not in active_codes:
        await update.message.reply_text("❌ Invalid or not-generated code. Ask admin for a valid code.")
        context.user_data["awaiting_code"] = False
        return
    del active_codes[code]
    persist_codes()
    context.user_data["awaiting_code"] = False
    if uid_int in current_round_queue:
        await update.message.reply_text("ℹ️ You are already in the current queue.")
        return
    current_round_queue.append(uid_int)
    await update.message.reply_text("🔥 You're in the queue! Waiting for other players...")
    logger.info("User %s joined queue using code %s (queue size=%d)", uid_int, code, len(current_round_queue))
    async with round_lock:
        if not round_running and len(current_round_queue) >= ROUND_PLAYER_COUNT:
            app.create_task(start_round_from_queue())

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📜 *Commands*\n"
        "`/register 07XXXXXXXX` — register with your phone\n"
        "`/entergame` — enter game (you'll be asked for code)\n"
        "`/claim` — claim your prize if you won\n\n"
        "*Admin only:*\n"
        "`/generatecode` — generate one one-time code\n"
        "`/listcodes` — list active codes\n"
        "`/forcestart` — force start a round\n"
    )
    await update.message.reply_markdown_v2(text)

async def cmd_claim(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global pending_payout
    user = update.effective_user
    uid_str = str(user.id)
    if not pending_payout:
        await update.message.reply_text("ℹ️ There is no pending payout at the moment.")
        return
    if pending_payout.get("user_id") != user.id:
        await update.message.reply_text("❌ You are not the pending winner.")
        return
    await safe_send(ADMIN_ID, f"✅ Payout claim by {make_username_display(players[uid_str])}\nPhone: {pending_payout.get('phone')}\nTaps: {pending_payout.get('taps')}")
    await update.message.reply_text("✅ Claim received. Admin has been notified to process your payout. 🎉")
    pending_payout = None
    persist_pending()

# -------------------- Admin commands --------------------
async def cmd_generatecode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user.id != ADMIN_ID:
        await update.message.reply_text("❌ You are not allowed to generate game codes.")
        return
    code = gen_code(8)
    while code in active_codes:
        code = gen_code(8)
    active_codes[code] = {"created": time.time(), "created_by": user.id}
    persist_codes()
    await update.message.reply_text(f"✅ Code generated: *{code}*\nShare it with one player only.", parse_mode="Markdown")
    logger.info("Admin %s generated code %s", user.id, code)

async def cmd_listcodes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user.id != ADMIN_ID:
        await update.message.reply_text("❌ Only admin can list codes.")
        return
    if not active_codes:
        await update.message.reply_text("No active codes.")
        return
    codes_list = "\n".join(active_codes.keys())
    await update.message.reply_text(f"Active codes:\n{codes_list}")

async def cmd_forcestart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user.id != ADMIN_ID:
        await update.message.reply_text("❌ Only admin can force-start rounds.")
        return
    async with round_lock:
        if round_running:
            await update.message.reply_text("❌ A round is already running.")
            return
        if not current_round_queue:
            await update.message.reply_text("❌ No queued players to start a round.")
            return
        app.create_task(start_round_from_queue())
        await update.message.reply_text("✅ Round force-started with queued players.")

# -------------------- Round orchestration --------------------
async def start_round_from_queue():
    global round_running, round_start_time, pending_payout, round_message_map
    async with round_lock:
        if round_running:
            logger.info("start_round_from_queue called but round already running.")
            return
        players_for_round: List[int] = current_round_queue[:ROUND_PLAYER_COUNT]
        del current_round_queue[:len(players_for_round)]
        round_running = True
        round_message_map = {}
        logger.info("Starting round with players: %s", players_for_round)
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        for uid in players_for_round:
            pdata = players.get(str(uid))
            if pdata:
                await safe_send(pdata["chat_id"], f"⏳ Game starting in *{i}*... Get ready! 🔥", parse_mode="Markdown")
        await asyncio.sleep(1)
    for uid in players_for_round:
        pdata = players.get(str(uid))
        if pdata:
            pdata["taps"] = 0
            pdata["taps_first5"] = 0
    persist_players()
    round_start_time = time.time()
    for uid in players_for_round:
        pdata = players.get(str(uid))
        if pdata:
            button = InlineKeyboardButton("👆 TAP", callback_data="tap")
            markup = InlineKeyboardMarkup([[button]])
            msg = await safe_send(pdata["chat_id"], f"🚀 Start tapping! {TAP_DURATION_SECONDS} seconds — TAP FAST! 💥", reply_markup=markup)
            if isinstance(msg, Message):
                round_message_map[uid] = msg.message_id
    await asyncio.sleep(TAP_DURATION_SECONDS)
    round_running = False
    snapshot_start = round_start_time
    round_start_time = None
    for uid, msg_id in list(round_message_map.items()):
        pdata = players.get(str(uid))
        if pdata:
            try:
                await app.bot.edit_message_reply_markup(chat_id=pdata["chat_id"], message_id=msg_id, reply_markup=None)
            except Exception:
                pass
    for uid in players_for_round:
        pdata = players.get(str(uid))
        if pdata:
            await safe_send(pdata["chat_id"], "⏱ Time's up! Tallying results... 🧾")
    participant_data = []
    for uid in players_for_round:
        pdata = players.get(str(uid))
        if pdata:
            participant_data.append({
                "user_id": uid,
                "display": make_username_display(pdata),
                "taps": int(pdata.get("taps", 0)),
                "taps_first5": int(pdata.get("taps_first5", 0)),
                "phone": pdata.get("phone"),
            })
        else:
            participant_data.append({"user_id": uid, "display": f"User{uid}", "taps": 0, "taps_first5": 0, "phone": None})
    participant_data_sorted = sorted(participant_data, key=lambda x: (x["taps"], x["taps_first5"]), reverse=True)
    winner = None
    if participant_data_sorted:
        top_taps = participant_data_sorted[0]["taps"]
        tied = [p for p in participant_data_sorted if p["taps"] == top_taps]
        if len(tied) == 1:
            winner = tied[0]
        else:
            winner = max(tied, key=lambda x: x["taps_first5"])
    lines = []
    rank = 1
    medals = ["🥇", "🥈", "🥉"]
    for p in participant_data_sorted:
        med = medals[rank - 1] if rank <= 3 else f"{rank}."
        display_safe = escape_markdown(p['display'], version=2)
        lines.append(f"{med} {display_safe} — {p['taps']} taps (first {p['taps_first5']}s)")
        rank += 1
    result_text = "🏁 ROUND RESULTS\n\n" + "\n".join(lines)
    if winner:
        winner_display_safe = escape_markdown(winner['display'], version=2)
        await safe_send(ADMIN_ID, f"🏆 Winner: {winner_display_safe}\n\n{result_text}", parse_mode="MarkdownV2")
    else:
        await safe_send(ADMIN_ID, f"No valid winner.\n\n{result_text}", parse_mode="MarkdownV2")
    for p in participant_data:
        pdata = players.get(str(p["user_id"]))
        if pdata:
            await safe_send(pdata["chat_id"], result_text, parse_mode="MarkdownV2")
    global pending_payout
    if winner:
        pending_payout = {
            "user_id": winner["user_id"],
            "phone": winner.get("phone"),
            "taps": winner["taps"],
            "expires": time.time() + PENDING_PAYOUT_SECONDS,
        }
        persist_pending()
        winner_pdata = players.get(str(winner["user_id"]))
        if winner_pdata:
            await safe_send(winner_pdata["chat_id"], "🎉 You won! Send /claim within 3 minutes to confirm your payout. 🏆")
        app.create_task(pending_payout_expirer())
    else:
        pending_payout = None

# -------------------- Pending payout watcher --------------------
async def pending_payout_expirer():
    global pending_payout
    while pending_payout:
        await asyncio.sleep(5)
        if time.time() > pending_payout["expires"]:
            uid = pending_payout["user_id"]
            pdata = players.get(str(uid))
            if pdata:
                await safe_send(pdata["chat_id"], "⏰ 3 minutes expired. You lost the prize. Better luck next time! 🍀")
            logger.info("Pending payout expired for user %s", uid)
            pending_payout = None
            persist_pending()
            break

# -------------------- Tap callback --------------------
async def callback_tap(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global round_start_time
    query = update.callback_query
    user = query.from_user
    uid_str = str(user.id)
    if uid_str not in players:
        await query.answer("You must register first with /register", show_alert=True)
        return
    if not round_running or round_start_time is None:
        await query.answer("Round not active!", show_alert=True)
        return
    elapsed = time.time() - round_start_time
    if elapsed > TAP_DURATION_SECONDS:
        await query.answer("Too late — round ended.", show_alert=True)
        return
    players[uid_str]["taps"] = int(players[uid_str].get("taps", 0)) + 1
    if elapsed <= 5:
        players[uid_str]["taps_first5"] = int(players[uid_str].get("taps_first5", 0)) + 1
    persist_players()
    await query.answer(f"Tapped! Total taps: {players[uid_str]['taps']}", show_alert=False)

# -------------------- Handlers registration --------------------
app.add_handler(CommandHandler("start", cmd_start))
app.add_handler(CommandHandler("help", cmd_help))
app.add_handler(CommandHandler("register", cmd_register))
app.add_handler(CommandHandler("entergame", cmd_entergame))
app.add_handler(CommandHandler("claim", cmd_claim))
app.add_handler(CommandHandler("generatecode", cmd_generatecode))
app.add_handler(CommandHandler("listcodes", cmd_listcodes))
app.add_handler(CommandHandler("forcestart", cmd_forcestart))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handler_text_codes))
app.add_handler(CallbackQueryHandler(callback_tap, pattern="tap"))

# -------------------- Bot start --------------------
if __name__ == "__main__":
    logger.info("Bot starting...")
    app.run_polling()
