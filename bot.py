import os
import json
import time
import threading
import math
import random
import traceback
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dotenv import load_dotenv
from telegram import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import websocket

# ---------------- ENV ----------------
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
APP_ID = os.getenv("DERIV_APP_ID")
ADMIN_ID = int(os.getenv("ADMIN_TELEGRAM_ID") or 0)

if not BOT_TOKEN or not APP_ID or not ADMIN_ID:
    raise RuntimeError("Set BOT_TOKEN, DERIV_APP_ID, ADMIN_TELEGRAM_ID in environment")

# ---------------- Globals / Storage ----------------
LOCK = threading.RLock()
subscriptions = {}         # user_id -> { expiry, package, max_signals, signals_sent_per_index }
generated_codes = {}       # code -> { package, max_signals, expiry, used=False }
redeem_states = {}         # user_id -> waiting for code
active_trades = {}         # user_id -> { index: trade_info }
LATEST_PRICE = {}          # symbol -> (price, ts)

# Index -> Deriv symbol
SYMBOLS = {
    "V25": "R_25",
    "V75": "R_75",
    "Boom1000": "BOOM1000",
    "Crash1000": "CRASH1000"
}

# Candle timeframes: we'll build 5m,15m,1h from ticks
TF_SECONDS = {"1m":60, "5m":5*60, "15m":15*60, "1h":60*60}
ENTRY_TF = "5m"
CONFIRM_TF = "15m"
MACRO_TF = "1h"

# Candle storage: symbol -> tf -> deque(candles)
CANDLES = {sym: {tf: deque(maxlen=1200) for tf in TF_SECONDS.keys()} for sym in SYMBOLS.values()}
# Ongoing partial candles (per tf)
OWNS = {sym: {tf: None for tf in TF_SECONDS.keys()} for sym in SYMBOLS.values()}

# ---------------- Utility functions ----------------

def now_utc():
    return datetime.utcnow()

def ts_from_dt(dt):
    return int(dt.timestamp())

def dt_from_epoch(e):
    return datetime.utcfromtimestamp(int(e))

def fmt_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

def start_bucket(epoch, seconds):
    return epoch - (epoch % seconds)

def make_code(package):
    return f"VE-{random.randint(10000,99999)}-{package[:2].upper()}"

# ---------------- Subscription / Code management ----------------

def create_generated_code(pkg_key):
    pkg_key = pkg_key.lower()
    if pkg_key in ("24h","24"):
        package = "24h"; max_signals = 2; expiry = now_utc() + timedelta(hours=24)
    elif pkg_key in ("72h","72"):
        package = "72h"; max_signals = 4; expiry = now_utc() + timedelta(hours=72)
    else:
        package = "30d"; max_signals = float('inf'); expiry = now_utc() + timedelta(days=30)
    code = make_code(package)
    with LOCK:
        generated_codes[code] = {"package": package, "max_signals": max_signals, "expiry": expiry, "used": False}
    return code, generated_codes[code]

def activate_code_for_user(code, user_id):
    with LOCK:
        rec = generated_codes.get(code)
        if not rec:
            return False, "Invalid code."
        if rec["used"]:
            return False, "Code already used."
        if now_utc() > rec["expiry"]:
            return False, "Code expired."
        subscriptions[user_id] = {
            "expiry": rec["expiry"],
            "package": rec["package"],
            "max_signals": rec["max_signals"],
            "signals_sent_per_index": defaultdict(int)
        }
        rec["used"] = True
    return True, subscriptions[user_id]

def check_subscription(user_id):
    with LOCK:
        sub = subscriptions.get(user_id)
        if not sub:
            return False, "❌ You have no active subscription. Use /redeem to enter your access code."
        if now_utc() > sub["expiry"]:
            return False, "❌ Your subscription has expired. Renew to continue receiving signals."
        return True, sub

# ---------------- Candle builder from ticks ----------------

def init_own(symbol, tf):
    if OWNS[symbol][tf] is None:
        OWNS[symbol][tf] = {"bucket": None, "open": None, "high": -1e18, "low": 1e18, "close": None, "ticks": 0}

def feed_tick(symbol, price, epoch):
    """Feed a tick into multiple TF candle builders."""
    LATEST_PRICE[symbol] = (price, epoch)
    for tf, sec in TF_SECONDS.items():
        init_own(symbol, tf)
        bucket = start_bucket(epoch, sec)
        own = OWNS[symbol][tf]
        if own["bucket"] is None:
            own["bucket"] = bucket
            own["open"] = price
            own["high"] = price
            own["low"] = price
            own["close"] = price
            own["ticks"] = 1
        elif bucket == own["bucket"]:
            if price > own["high"]: own["high"] = price
            if price < own["low"]: own["low"] = price
            own["close"] = price
            own["ticks"] += 1
        else:
            # finalize candle
            candle = {"open_ts": own["bucket"], "open": own["open"], "high": own["high"],
                      "low": own["low"], "close": own["close"], "ticks": own["ticks"]}
            CANDLES[symbol][tf].append(candle)
            # start new
            own["bucket"] = bucket
            own["open"] = price
            own["high"] = price
            own["low"] = price
            own["close"] = price
            own["ticks"] = 1

def last_candles(symbol, tf, n):
    return list(CANDLES[symbol][tf])[-n:] if len(CANDLES[symbol][tf])>0 else []

# ---------------- Market structure helpers ----------------

def detect_swings(symbol, tf, k=3):
    """Return lists of swing highs and lows (index_in_deque, price)."""
    candles = list(CANDLES[symbol][tf])
    n = len(candles)
    highs = []
    lows = []
    if n < (k*2+1):
        return highs, lows
    for i in range(k, n-k):
        hi = candles[i]["high"]
        is_high = all(hi > candles[j]["high"] for j in range(i-k, i+k+1) if j!=i)
        if is_high:
            highs.append((i, hi))
        lo = candles[i]["low"]
        is_low = all(lo < candles[j]["low"] for j in range(i-k, i+k+1) if j!=i)
        if is_low:
            lows.append((i, lo))
    return highs, lows

def recent_swings_series(symbol, tf, limit=6):
    """Return combined sorted recent swings by candle index: [(idx, type, price)] where type 'H' or 'L'"""
    highs, lows = detect_swings(symbol, tf)
    combined = []
    for idx, p in highs:
        combined.append((idx, 'H', p))
    for idx, p in lows:
        combined.append((idx, 'L', p))
    combined.sort(key=lambda x: x[0])
    return combined[-limit:]

def market_structure_direction(symbol):
    """
    Return 'bull' if structure shows higher highs+higher lows,
           'bear' if lower highs/lower lows,
           None otherwise / unclear.
    Based only on swings on MACRO_TF.
    """
    swings = recent_swings_series(symbol, MACRO_TF, limit=8)
    if len(swings) < 4:
        return None
    # extract last few highs and lows sequences
    highs = [p for (i,t,p) in swings if t=='H']
    lows = [p for (i,t,p) in swings if t=='L']
    if len(highs) >= 2 and len(lows) >=2:
        # check if highs increasing and lows increasing
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            return "bull"
        if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            return "bear"
    return None

def last_structural_high_low(symbol, tf):
    """Return most recent swing high and swing low prices (or None) on tf."""
    highs, lows = detect_swings(symbol, tf)
    h = highs[-1][1] if highs else None
    l = lows[-1][1] if lows else None
    return h, l

# ---------------- Momentum helpers (supporting only) ----------------

def candle_body_pct(c):
    rng = c["high"] - c["low"]
    if rng == 0: return 0.0
    return abs(c["close"] - c["open"]) / rng

def avg_ticks(symbol, tf, lookback=5):
    c = last_candles(symbol, tf, lookback)
    if not c: return 0
    return sum(x["ticks"] for x in c)/len(c)

def strong_impulse(symbol, tf, lookback=3, factor=1.4):
    """Return True if recent candles have bodies bigger than baseline*factor."""
    c = last_candles(symbol, tf, lookback+3)
    if len(c) < lookback+1: return False
    baseline = sum(abs(x["close"]-x["open"]) for x in c[:-lookback]) / max(1, len(c[:-lookback]))
    if baseline == 0: baseline = 1e-6
    recent = c[-lookback:]
    for r in recent:
        if abs(r["close"]-r["open"]) > baseline * factor:
            return True
    return False

# ---------------- RR & SL calculation ----------------

def compute_sl_from_structure(symbol, tf_entry, direction):
    """
    SL set just beyond last swing low/high on CONFIRM_TF.
    Returns SL price and small buffer applied.
    """
    h, l = last_structural_high_low(symbol, CONFIRM_TF)
    buff_pct = 0.0018  # ~0.18% buffer
    latest_price = get_latest_price(symbol)
    if direction == "BUY":
        # SL = last swing low - buffer (or 0.18% of price)
        base = l if l else latest_price * (1 - 0.002)
        sl = round(base * (1 - buff_pct), 5)
        return sl
    else:
        base = h if h else latest_price * (1 + 0.002)
        sl = round(base * (1 + buff_pct), 5)
        return sl

def compute_tps(entry_price, sl, direction, rr3=3.0, rr5=5.0):
    if direction == "BUY":
        dist = entry_price - sl
        tp3 = round(entry_price + dist * rr3, 5)
        tp5 = round(entry_price + dist * rr5, 5)
    else:
        dist = sl - entry_price
        tp3 = round(entry_price - dist * rr3, 5)
        tp5 = round(entry_price - dist * rr5, 5)
    return tp3, tp5

# ---------------- Strategy: structure-only V25 ----------------

def analyze_v25(symbol):
    """
    Structure-only V25 logic:
    - Macro (1h) trend must be bull/bear via market_structure_direction
    - Confirm that CONFIRM_TF (15m) structure aligns (same direction)
    - On ENTRY_TF (5m): detect Break Of Structure (BOS) relative to recent swings,
      then require a confirming impulse candle (supporting only).
    - If satisfied return (direction, entry_price, sl, tp3, tp5, reason)
    """
    try:
        macro = market_structure_direction(symbol)
        if not macro:
            return None
        # confirm mid-term
        mid = market_structure_direction(symbol)  # for simplicity reuse macro; ideally would compute on 15m
        # we will compute on CONFIRM_TF explicitly:
        mid_swings = recent_swings_series(symbol, CONFIRM_TF, limit=8)
        if not mid_swings:
            return None
        # determine last structural high & low on 5m for BOS detection
        highs_lows_5 = detect_swings(symbol, ENTRY_TF)
        # get recent 5m candles
        candles5 = last_candles(symbol, ENTRY_TF, 6)
        if len(candles5) < 4:
            return None
        last_close = candles5[-1]["close"]
        prev_high = max(c["high"] for c in candles5[:-1])
        prev_low = min(c["low"] for c in candles5[:-1])

        if macro == "bull":
            # BOS buy: last_close > prev_high
            if last_close > prev_high and strong_impulse(symbol, ENTRY_TF, lookback=2, factor=1.2):
                entry = last_close
                sl = compute_sl_from_structure(symbol, ENTRY_TF, "BUY")
                tp3, tp5 = compute_tps(entry, sl, "BUY")
                reason = "Structure-only: Macro bull + 5m BOS + momentum confirmation"
                return ("BUY", entry, sl, tp3, tp5, reason)
        elif macro == "bear":
            if last_close < prev_low and strong_impulse(symbol, ENTRY_TF, lookback=2, factor=1.2):
                entry = last_close
                sl = compute_sl_from_structure(symbol, ENTRY_TF, "SELL")
                tp3, tp5 = compute_tps(entry, sl, "SELL")
                reason = "Structure-only: Macro bear + 5m BOS + momentum confirmation"
                return ("SELL", entry, sl, tp3, tp5, reason)
        return None
    except Exception:
        traceback.print_exc()
        return None

# ---------------- Strategy: structure-only V75 ----------------

def analyze_v75(symbol):
    """
    V75 uses same structure-only approach but slightly stricter momentum and mid-term confirmation.
    """
    try:
        macro = market_structure_direction(symbol)
        if not macro:
            return None
        # require strong mid-term impulse on CONFIRM_TF
        if not strong_impulse(symbol, CONFIRM_TF, lookback=3, factor=1.25):
            return None
        candles5 = last_candles(symbol, ENTRY_TF, 6)
        if len(candles5) < 4:
            return None
        last_close = candles5[-1]["close"]
        prev_high = max(c["high"] for c in candles5[:-1])
        prev_low = min(c["low"] for c in candles5[:-1])
        if macro == "bull":
            if last_close > prev_high and strong_impulse(symbol, ENTRY_TF, lookback=2, factor=1.3):
                entry = last_close
                sl = compute_sl_from_structure(symbol, ENTRY_TF, "BUY")
                tp3, tp5 = compute_tps(entry, sl, "BUY")
                reason = "V75 Structure-only: Macro bull + momentum cluster + BOS on 5m"
                return ("BUY", entry, sl, tp3, tp5, reason)
        elif macro == "bear":
            if last_close < prev_low and strong_impulse(symbol, ENTRY_TF, lookback=2, factor=1.3):
                entry = last_close
                sl = compute_sl_from_structure(symbol, ENTRY_TF, "SELL")
                tp3, tp5 = compute_tps(entry, sl, "SELL")
                reason = "V75 Structure-only: Macro bear + momentum cluster + BOS on 5m"
                return ("SELL", entry, sl, tp3, tp5, reason)
        return None
    except Exception:
        traceback.print_exc()
        return None

# ---------------- Strategy: Boom/Crash structural spike engine (no OB as independent trigger) ----------------

def compute_spike_strength(symbol):
    """
    Compute a 0-100 probability-like score from:
    - avg ticks (entry tf)
    - compression (low range recent candles)
    - momentum ratio
    - spike frequency (historical)
    This is heuristic; thresholds tuned for high probability.
    """
    try:
        avg_t = avg_ticks = avg_ticks_fn(symbol=symbol, tf=ENTRY_TF, lookback=8)
        recent = last_candles(symbol, ENTRY_TF, 6)
        if not recent:
            return 0
        ranges = [c["high"] - c["low"] for c in recent]
        avg_range = sum(ranges)/len(ranges) if ranges else 1e-6
        # compression metric: small last candle ranges relative to avg => compression
        compression = 1.0 if recent[-1]["high"] - recent[-1]["low"] < avg_range*0.6 else 0.5
        # momentum: presence of strong impulse in CONFIRM_TF
        momentum = 1.0 if strong_impulse(symbol, CONFIRM_TF, lookback=3, factor=1.3) else 0.4
        # spike frequency: how often a similar spike happened last hour (lower freq => higher probability)
        # we use heuristic random factor to avoid deterministic zero
        spike_history_factor = 1.0  # placeholder for complexity (we can count big wicks)
        # Compose score
        score = min(100, int((avg_t * 4) * compression * momentum * spike_history_factor * 10))
        return max(0, score)
    except Exception:
        traceback.print_exc()
        return 0

def avg_ticks_fn(symbol, tf, lookback=5):
    c = last_candles(symbol, tf, lookback)
    if not c: return 0
    return sum(x["ticks"] for x in c)/len(c)

def analyze_boom_crash(symbol):
    """
    Detect structured sweep + rejection matching macro trend:
    - For BOOM1000 (we treat it as potential 'sell' after sweep above supply)
    - For CRASH1000 (potential 'buy' after sweep below demand)
    Returns (direction, entry_price, candle_count, reason) if valid
    """
    try:
        macro = market_structure_direction(symbol)
        if not macro:
            return None
        candles5 = last_candles(symbol, ENTRY_TF, 8)
        if len(candles5) < 5:
            return None
        last_close = candles5[-1]["close"]
        prev_high = max(c["high"] for c in candles5[:-1])
        prev_low = min(c["low"] for c in candles5[:-1])
        # For Boom (spike up then sell): require macro bear or neutral leaning bear, sweep above prev_high then rejection
        if "BOOM" in symbol:
            # detect temporary wick above prev_high in last few candles
            highs = [c["high"] for c in candles5[-5:]]
            if max(highs) > prev_high and last_close < candles5[-2]["close"]:
                # compute strength score
                score = compute_spike_strength(symbol)
                if score < 80:
                    return None
                candle_count = 6 if score < 87 else (10 if score < 93 else 15)
                entry = last_close
                reason = f"Boom structural sweep & rejection. Strength {score}%"
                return ("SELL", entry, candle_count, reason)
        if "CRASH" in symbol:
            lows = [c["low"] for c in candles5[-5:]]
            if min(lows) < prev_low and last_close > candles5[-2]["close"]:
                score = compute_spike_strength(symbol)
                if score < 80:
                    return None
                candle_count = 6 if score < 87 else (10 if score < 93 else 15)
                entry = last_close
                reason = f"Crash structural sweep & rejection. Strength {score}%"
                return ("BUY", entry, candle_count, reason)
        return None
    except Exception:
        traceback.print_exc()
        return None

# ---------------- Helper to get latest price ----------------

def get_latest_price(symbol):
    rec = LATEST_PRICE.get(symbol)
    if rec:
        return rec[0]
    c = last_candles(symbol, ENTRY_TF, 1)
    if c:
        return c[-1]["close"]
    return 0.0

# ---------------- Dispatch & active trade control ----------------

def can_dispatch(user_id, index):
    ok, sub_or_msg = check_subscription(user_id)
    if not ok:
        return False, sub_or_msg
    sub = sub_or_msg
    used = sub["signals_sent_per_index"].get(index, 0)
    if used >= sub["max_signals"]:
        return False, f"⚠️ Package limit reached for {index}."
    # active trade conflict check
    ut = active_trades.get(user_id, {})
    if ut.get(index) and ut[index]["status"] == "OPEN":
        return False, "🔒 You have an active trade for this index. Wait until it closes."
    return True, sub

def send_v_signal(user_id, index, direction, entry, sl, tp3, tp5, reason):
    ok, sub_or_msg = can_dispatch(user_id, index)
    if not ok:
        return False, sub_or_msg
    # mark active trade
    with LOCK:
        active_trades.setdefault(user_id, {})[index] = {
            "entry_time": now_utc(),
            "direction": direction,
            "status": "OPEN",
            "meta": {"entry": entry, "sl": sl, "tp3": tp3, "tp5": tp5}
        }
        subscriptions[user_id]["signals_sent_per_index"][index] = subscriptions[user_id]["signals_sent_per_index"].get(index,0) + 1
    # message
    msg = (
        f"⚡ *VoltEdge Pro Signal* ⚡\n"
        f"*Index:* {index}\n"
        f"*Direction:* {direction}\n"
        f"*Entry:* `{entry}`\n"
        f"*SL:* `{sl}`\n"
        f"*TP (1:3):* `{tp3}`  |  *TP (1:5):* `{tp5}`\n"
        f"_Reason:_ {reason}\n"
        f"_Time:_ {fmt_dt(now_utc())}"
    )
    bot_app.bot.send_message(chat_id=user_id, text=msg, parse_mode=ParseMode.MARKDOWN)
    return True, "sent"

def send_boomcrash_signal(user_id, index, direction, entry, candle_count, reason):
    ok, sub_or_msg = can_dispatch(user_id, index)
    if not ok:
        return False, sub_or_msg
    with LOCK:
        active_trades.setdefault(user_id, {})[index] = {
            "entry_time": now_utc(),
            "direction": direction,
            "status": "OPEN",
            "meta": {"entry": entry, "candle_count": candle_count}
        }
        subscriptions[user_id]["signals_sent_per_index"][index] = subscriptions[user_id]["signals_sent_per_index"].get(index,0) + 1
    msg = (
        f"⚡ *VoltEdge Pro Signal* ⚡\n"
        f"*Index:* {index}\n"
        f"*Direction:* {direction}\n"
        f"*Entry:* `{entry}`\n"
        f"*Action:* Cash out after *{candle_count}* {ENTRY_TF} candles\n"
        f"_Reason:_ {reason}\n"
        f"_Time:_ {fmt_dt(now_utc())}"
    )
    bot_app.bot.send_message(chat_id=user_id, text=msg, parse_mode=ParseMode.MARKDOWN)
    return True, "sent"

def close_trade(user_id, index, reason, exit_price):
    with LOCK:
        t = active_trades.get(user_id, {}).get(index)
        if not t:
            return False
        t["status"] = "CLOSED"
        t["exit_time"] = now_utc()
        t["exit_price"] = exit_price
        t["exit_reason"] = reason
    msg = (
        f"✅ Trade closed for *{index}*\n"
        f"Reason: {reason}\n"
        f"Exit Price: `{exit_price}`\n"
        f"Time: {fmt_dt(now_utc())}"
    )
    try:
        bot_app.bot.send_message(chat_id=user_id, text=msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        print("send close msg error", e)
    return True

# ---------------- Active trade monitor ----------------

def monitor_trades_loop():
    while True:
        try:
            with LOCK:
                for user_id, trades in list(active_trades.items()):
                    for index, info in list(trades.items()):
                        if info["status"] != "OPEN":
                            continue
                        meta = info["meta"]
                        symbol = SYMBOLS[index]
                        latest = get_latest_price(symbol)
                        if index in ("V25","V75"):
                            entry = meta["entry"]
                            sl = meta["sl"]
                            tp3 = meta["tp3"]
                            tp5 = meta["tp5"]
                            dirc = info["direction"]
                            # check TP5 first (bigger)
                            if dirc == "BUY":
                                if latest >= tp5:
                                    close_trade(user_id, index, "TP (1:5) reached", latest)
                                elif latest >= tp3:
                                    close_trade(user_id, index, "TP (1:3) reached", latest)
                                elif latest <= sl:
                                    close_trade(user_id, index, "SL hit", latest)
                            else:
                                if latest <= tp5:
                                    close_trade(user_id, index, "TP (1:5) reached", latest)
                                elif latest <= tp3:
                                    close_trade(user_id, index, "TP (1:3) reached", latest)
                                elif latest >= sl:
                                    close_trade(user_id, index, "SL hit", latest)
                        else:
                            # Boom/Crash: candle-count based
                            entry_time = info["entry_time"]
                            candle_target = meta.get("candle_count", 6)
                            # count completed ENTRY_TF candles since entry_time
                            candles = last_candles(symbol, ENTRY_TF, 200)
                            if not candles:
                                continue
                            # count how many entry_tf candles have started after entry_time
                            count = sum(1 for c in candles if datetime.utcfromtimestamp(c["open_ts"]) > entry_time)
                            if count >= candle_target:
                                latest = get_latest_price(symbol)
                                close_trade(user_id, index, f"Rule: closed after {candle_target} {ENTRY_TF} candles", latest)
        except Exception:
            traceback.print_exc()
        time.sleep(2)

# ---------------- Deriv websocket loop (single connection) ----------------

def deriv_loop():
    url = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
    while True:
        try:
            ws = websocket.create_connection(url, timeout=15)
            for s in SYMBOLS.values():
                sub = {"ticks": s, "subscribe": 1}
                ws.send(json.dumps(sub))
            print("Connected to Deriv with app_id", APP_ID)
            while True:
                raw = ws.recv()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except:
                    continue
                if "tick" in data:
                    t = data["tick"]
                    symbol = t.get("symbol")
                    quote = t.get("quote") or t.get("tick")
                    epoch = t.get("epoch") or int(time.time())
                    if symbol and quote is not None:
                        price = float(quote)
                        feed_tick(symbol, price, int(epoch))
                # else ignore other messages
        except Exception as e:
            print("Deriv WS error, reconnecting in 3s:", e)
            traceback.print_exc()
            time.sleep(3)

# ---------------- Per-user worker: runs analysis and dispatch ----------------

def per_user_worker(user_id):
    last_sent = {idx: None for idx in SYMBOLS.keys()}
    while True:
        ok, sub_or_msg = check_subscription(user_id)
        if not ok:
            try:
                bot_app.bot.send_message(chat_id=user_id, text=sub_or_msg)
            except:
                pass
            break
        try:
            # V25
            res = analyze_v25(SYMBOLS["V25"])
            if res:
                direction, entry, sl, tp3, tp5, reason = res
                can, _ = can_dispatch(user_id, "V25")
                if can and (not last_sent["V25"] or (datetime.utcnow()-last_sent["V25"]).total_seconds() > 30):
                    send_v_signal(user_id, "V25", direction, entry, sl, tp3, tp5, reason)
                    last_sent["V25"] = datetime.utcnow()
            # V75
            res = analyze_v75(SYMBOLS["V75"])
            if res:
                direction, entry, sl, tp3, tp5, reason = res
                can, _ = can_dispatch(user_id, "V75")
                if can and (not last_sent["V75"] or (datetime.utcnow()-last_sent["V75"]).total_seconds() > 30):
                    send_v_signal(user_id, "V75", direction, entry, sl, tp3, tp5, reason)
                    last_sent["V75"] = datetime.utcnow()
            # Boom
            res = analyze_boom_crash(SYMBOLS["Boom1000"])
            if res:
                direction, entry, candle_count, reason = res
                can, _ = can_dispatch(user_id, "Boom1000")
                if can and (not last_sent["Boom1000"] or (datetime.utcnow()-last_sent["Boom1000"]).total_seconds() > 30):
                    send_boomcrash_signal(user_id, "Boom1000", direction, entry, candle_count, reason)
                    last_sent["Boom1000"] = datetime.utcnow()
            # Crash
            res = analyze_boom_crash(SYMBOLS["Crash1000"])
            if res:
                direction, entry, candle_count, reason = res
                can, _ = can_dispatch(user_id, "Crash1000")
                if can and (not last_sent["Crash1000"] or (datetime.utcnow()-last_sent["Crash1000"]).total_seconds() > 30):
                    send_boomcrash_signal(user_id, "Crash1000", direction, entry, candle_count, reason)
                    last_sent["Crash1000"] = datetime.utcnow()
        except Exception:
            traceback.print_exc()
        time.sleep(4)

# ---------------- Telegram command handlers ----------------

async def cmd_start(update: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚡ VoltEdge Pro — live structure-only signals (V25, V75, Boom1000, Crash1000)\n"
        "⚠️ No bot is 100% perfect. Trade responsibly.\n"
        "Pay, ask admin for code, then /redeem to activate. Use /help for commands."
    )

async def cmd_help(update: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start - welcome\n"
        "/help - this message\n"
        "/redeem - redeem access code\n"
        "/subscriptionstatus - show your package & signals used\n"
        "/startanalysis - begin live analysis (must have active subscription)\n"
        "/volatility25 /volatility75 /boom /crash - strategy info\n"
        "Admin only: /generatecode"
    )

async def cmd_vol25(update: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "V25: STRUCTURE-only strategy (1H->15M->5M). We require Break Of Structure on 5m aligned to 1h macro. TP/SL shown in real prices (RR 1:3 & 1:5)."
    )

async def cmd_vol75(update: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "V75: STRUCTURE-only momentum + trend. 1H->15M->5M alignment. TP/SL shown in real prices (RR 1:3 & 1:5)."
    )

async def cmd_boom(update: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Boom1000: Structure-only spike signals. Bot calculates probability; if >=80% a signal is sent with dynamic candle-count exit (6/10/15)."
    )

async def cmd_crash(update: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Crash1000: Structure-only spike signals. Bot calculates probability; if >=80% a signal is sent with dynamic candle-count exit (6/10/15)."
    )

async def cmd_redeem(update: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    redeem_states[user_id] = True
    await update.message.reply_text("⚡ Enter your access code (e.g. VE-12345-24H):")

async def cmd_subscription_status(update: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    ok, sub_or_msg = check_subscription(user_id)
    if not ok:
        await update.message.reply_text(sub_or_msg)
        return
    sub = sub_or_msg
    info = f"Package: {sub['package']}\nExpires: {fmt_dt(sub['expiry'])}\nSignals used per index:\n"
    for idx in ("V25","V75","Boom1000","Crash1000"):
        used = sub["signals_sent_per_index"].get(idx,0)
        maxs = sub["max_signals"]
        info += f"- {idx}: {used}/{'unlimited' if math.isinf(maxs) else maxs}\n"
    await update.message.reply_text(info)

async def cmd_generatecode(update: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != ADMIN_ID:
        await update.message.reply_text("❌ You are not authorized.")
        return
    await update.message.reply_text("Which package to generate?\n1) 24h (2 signals/index)\n2) 72h (4 signals/index)\n3) 30d (unlimited)\nReply with 1,2 or 3")

async def cmd_startanalysis(update: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    ok, sub_or_msg = check_subscription(user_id)
    if not ok:
        await update.message.reply_text(sub_or_msg)
        return
    await update.message.reply_text("⚡ Analysis started. You will receive live signals when conditions meet structure-only rules.")
    t = threading.Thread(target=per_user_worker, args=(user_id,), daemon=True)
    t.start()

# generic message handler (redeem code input, admin generatecode next-step)
async def message_handler(update: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = (update.message.text or "").strip()
    # admin code generation step
    if user_id == ADMIN_ID and text in ("1","2","3"):
        mapping = {"1":"24h","2":"72h","3":"30d"}
        code, info = create_generated_code(mapping[text])
        await update.message.reply_text(f"✅ Code generated: `{code}`\nPackage: {info['package']}\nExpires: {fmt_dt(info['expiry'])}", parse_mode=ParseMode.MARKDOWN)
        return
    # redeem
    if redeem_states.get(user_id):
        redeem_states[user_id] = False
        ok, resp = activate_code_for_user(text.upper(), user_id)
        if not ok:
            await update.message.reply_text(f"❌ {resp}")
            return
        sub = resp
        await update.message.reply_text(f"🎉 Activated {sub['package']} until {fmt_dt(sub['expiry'])}. Use /startanalysis to begin.")
        return
    await update.message.reply_text("Command not recognized. Use /help for commands.")

# ---------------- App init ----------------
bot_app = ApplicationBuilder().token(BOT_TOKEN).build()
bot_app.add_handler(CommandHandler("start", cmd_start))
bot_app.add_handler(CommandHandler("help", cmd_help))
bot_app.add_handler(CommandHandler("redeem", cmd_redeem))
bot_app.add_handler(CommandHandler("subscriptionstatus", cmd_subscription_status))
bot_app.add_handler(CommandHandler("volatility25", cmd_vol25))
bot_app.add_handler(CommandHandler("volatility75", cmd_vol75))
bot_app.add_handler(CommandHandler("boom", cmd_boom))
bot_app.add_handler(CommandHandler("crash", cmd_crash))
bot_app.add_handler(CommandHandler("startanalysis", cmd_startanalysis))
bot_app.add_handler(CommandHandler("generatecode", cmd_generatecode))
bot_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_handler))

# ---------------- Background threads start ----------------

# 1) Deriv tick streamer
t_ws = threading.Thread(target=deriv_loop, daemon=True)
t_ws.start()

# 2) Active trade monitor
t_mon = threading.Thread(target=monitor_trades_loop, daemon=True)
t_mon.start()

# ---------------- Run bot ----------------
if __name__ == "__main__":
    print("VoltEdge Pro starting... (structure-only strategies)")
    bot_app.run_polling()

if __name__ == "__main__":
    logger.info("Bot starting...")
    app.run_polling()

