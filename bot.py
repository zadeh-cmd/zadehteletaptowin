import os
import asyncio
import time
import random
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

BOT_TOKEN = os.getenv("BOT_TOKEN")

app = ApplicationBuilder().token(BOT_TOKEN).build()

players = {}  # username -> {phone, taps, first5}
round_active = False
start_time = None

async def register(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Register players with Mpesa numbers"""
    if len(context.args) != 1:
        await update.message.reply_text("Use: /register 07XXXXXXXX")
        return

    phone = context.args[0]
    username = update.effective_user.username
    players[username] = {"phone": phone, "taps": 0, "first5": 0}
    await update.message.reply_text(f"✅ Registered @{username} ({phone})")

async def startgame(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start tapping round (admin only)"""
    global round_active, start_time
    chat_id = update.message.chat_id
    admins = [admin.user.id for admin in await context.bot.get_chat_administrators(chat_id)]

    if update.effective_user.id not in admins:
        await update.message.reply_text("❌ Only admins can start the game.")
        return

    if round_active:
        await update.message.reply_text("A round is already running!")
        return

    if len(players) < 2:
        await update.message.reply_text("Need at least 2 registered players to start.")
        return

    round_active = True
    start_time = time.time()
    await update.message.reply_text("🎮 Game starts in 5 seconds...")
    await asyncio.sleep(5)

    button = InlineKeyboardButton("👆 TAP", callback_data="tap")
    markup = InlineKeyboardMarkup([[button]])
    msg = await update.message.reply_text("🚀 Start tapping! 10 seconds go!", reply_markup=markup)

    await asyncio.sleep(10)
    round_active = False
    await msg.edit_text("⏱ Time’s up! Calculating results...")

    if not players:
        await update.message.reply_text("No registered players.")
        return

    # find highest tapper
    highest = max(p["taps"] for p in players.values())
    tied = [u for u, d in players.items() if d["taps"] == highest]

    if len(tied) == 1:
        winner = tied[0]
    else:
        # if tie → who reached score earliest (first5)
        winner = min(tied, key=lambda u: players[u]["first5"])

    w_phone = players[winner]["phone"]
    w_taps = players[winner]["taps"]

    result = "\n".join([f"@{u}: {d['taps']} taps" for u, d in players.items()])
    await update.message.reply_text(
        f"🏆 Winner: @{winner}\n📞 Mpesa: {w_phone}\n💥 Taps: {w_taps}\n\n📊 Results:\n{result}"
    )

    # 🧹 Remove all non-admin players
    await update.message.reply_text("🧹 Clearing players... next round coming soon!")
    chat = await context.bot.get_chat(chat_id)
    admins = [admin.user.id for admin in await chat.get_administrators()]
    async for member in context.bot.get_chat_administrators(chat_id):
        pass  # just get admin list

    async for member in context.bot.get_chat_members(chat_id):
        if member.user.id not in admins and not member.user.is_bot:
            try:
                await context.bot.ban_chat_member(chat_id, member.user.id)
                await context.bot.unban_chat_member(chat_id, member.user.id)
            except Exception:
                pass

    # reset for next round
    for p in players.values():
        p["taps"] = 0
        p["first5"] = 0

async def tap(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Track tap counts"""
    global start_time
    if not round_active or start_time is None:
        await update.callback_query.answer("Round not active!", show_alert=True)
        return

    username = update.effective_user.username
    if username not in players:
        await update.callback_query.answer("You must register first with /register", show_alert=True)
        return

    elapsed = time.time() - start_time
    players[username]["taps"] += 1
    if elapsed <= 5:
        players[username]["first5"] = elapsed
    await update.callback_query.answer(f"Tap count: {players[username]['taps']}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/register 07XXXXXXXX — register with phone\n/startgame — admin starts round"
    )

app.add_handler(CommandHandler("register", register))
app.add_handler(CommandHandler("startgame", startgame))
app.add_handler(CommandHandler("help", help_command))
app.add_handler(CallbackQueryHandler(tap, pattern="^tap$"))

if __name__ == "__main__":
    app.run_polling()
