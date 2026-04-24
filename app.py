import streamlit as st
import yfinance as yf
import google.generativeai as genai
import feedparser
import smtplib
import re
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Advisor (India)", layout="wide", page_icon="📈")

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("🔑 Add GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()


# ── WISHLIST FILE ─────────────────────────────────────────────────────────────
WISHLIST_FILE = "wishlist.json"

def load_wishlist() -> list[dict]:
    try:
        if os.path.exists(WISHLIST_FILE):
            with open(WISHLIST_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_wishlist(wishlist: list[dict]):
    try:
        with open(WISHLIST_FILE, "w") as f:
            json.dump(wishlist, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save wishlist: {e}")


# ── SESSION STATE INIT ────────────────────────────────────────────────────────
if "wishlist"      not in st.session_state:
    st.session_state.wishlist      = load_wishlist()
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "auto_ticker"   not in st.session_state:
    st.session_state.auto_ticker   = None
if "auto_name"     not in st.session_state:
    st.session_state.auto_name     = None


# ── AUTO-DETECT BEST MODEL ────────────────────────────────────────────────────
def get_best_model() -> str:
    preferred = [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    try:
        available = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        for model in preferred:
            if model in available:
                return model
    except Exception:
        pass
    return "gemini-1.5-flash"

MODEL = get_best_model()


# ── HELPERS ───────────────────────────────────────────────────────────────────
def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = (text.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&nbsp;", " ")
                .replace("&quot;", '"'))
    return text.strip()


def clean_ticker(raw: str, suffix: str) -> str | None:
    result = raw.strip().upper()
    result = result.split("\n")[0].strip()
    result = result.replace("'", "").replace('"', "").replace(" ", "")
    while ".." in result:
        result = result.replace("..", ".")
    result = result.rstrip(".,;:")
    if not result or result == "UNKNOWN":
        return None
    base = result.replace(".NS", "").replace(".BO", "")
    if len(base) < 2:
        return None
    if not (result.endswith(".NS") or result.endswith(".BO")):
        result = result.split()[0] + suffix
    return result


def add_to_wishlist(name: str, ticker: str, price: float) -> bool:
    already = any(w["ticker"] == ticker for w in st.session_state.wishlist)
    if not already:
        st.session_state.wishlist.append({
            "name":   name,
            "ticker": ticker,
            "price":  price,
            "added":  datetime.now().strftime("%d %b %Y %I:%M %p"),
        })
        save_wishlist(st.session_state.wishlist)
        return True
    return False


# ── TICKER RESOLVER ───────────────────────────────────────────────────────────
def resolve_ticker(company_name: str, exchange: str = "NSE") -> str | None:
    suffix = ".NS" if exchange == "NSE" else ".BO"
    prompt = f"""You are a stock ticker database for Indian stock markets.
Task: Find the exact NSE/BSE ticker symbol for the company below.
Company: {company_name}
Exchange: {exchange}
Required suffix: {suffix}
Rules:
- Reply with ONLY the complete ticker symbol including the suffix
- Do not truncate or shorten the ticker
- Do not add any explanation or extra text
Examples: WIPRO.NS RELIANCE.NS HDFCBANK.NS TCS.NS INFY.NS SBIN.NS ZOMATO.NS
If truly unknown reply: UNKNOWN
Complete ticker symbol for {company_name}:"""
    try:
        model    = genai.GenerativeModel(MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=200, temperature=0.0),
        )
        result = clean_ticker(response.text, suffix)
        if result:
            base = result.replace(".NS", "").replace(".BO", "")
            if len(base) < 3:
                retry = model.generate_content(
                    f"Write the FULL NSE ticker for {company_name} ending in "
                    f"{suffix}. Example: Wipro = WIPRO.NS. Just the ticker:",
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=200, temperature=0.0),
                )
                result = clean_ticker(retry.text, suffix)
        return result
    except Exception as e:
        st.warning(f"Ticker lookup error: {e}")
        return None


# ── STOCK DATA ────────────────────────────────────────────────────────────────
def fetch_stock_data(ticker: str) -> dict | None:
    try:
        asset = yf.Ticker(ticker)
        hist  = asset.history(period="3mo")
        if hist.empty:
            return None
        curr     = round(float(hist["Close"].iloc[-1]), 2)
        prev     = round(float(hist["Close"].iloc[-2]), 2)
        high_52w = round(float(hist["High"].max()), 2)
        low_52w  = round(float(hist["Low"].min()), 2)
        sma20    = round(float(hist["Close"].tail(20).mean()), 2)
        sma50    = (round(float(hist["Close"].tail(50).mean()), 2)
                    if len(hist) >= 50 else None)
        change     = round(curr - prev, 2)
        change_pct = round((change / prev) * 100, 2)
        avg_vol    = int(hist["Volume"].tail(20).mean())
        last_vol   = int(hist["Volume"].iloc[-1])
        return {
            "ticker": ticker, "price": curr, "change": change,
            "change_pct": change_pct, "high_52w": high_52w,
            "low_52w": low_52w, "sma20": sma20, "sma50": sma50,
            "avg_vol": avg_vol, "last_vol": last_vol, "hist": hist,
        }
    except Exception as e:
        st.error(f"Price fetch error: {e}")
        return None


# ── TOP GAINERS / LOSERS ──────────────────────────────────────────────────────
NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","HCLTECH.NS",
    "WIPRO.NS","SUNPHARMA.NS","TITAN.NS","BAJFINANCE.NS","NESTLEIND.NS",
    "ULTRACEMCO.NS","TECHM.NS","POWERGRID.NS","NTPC.NS","ONGC.NS",
    "TATAMOTORS.NS","TATASTEEL.NS","JSWSTEEL.NS","ADANIENT.NS","ADANIPORTS.NS",
    "COALINDIA.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","EICHERMOT.NS","DIVISLAB.NS",
    "DRREDDY.NS","CIPLA.NS","APOLLOHOSP.NS","SBILIFE.NS","HDFCLIFE.NS",
    "HINDALCO.NS","VEDL.NS","BPCL.NS","IOC.NS","GRASIM.NS",
    "TATACONSUM.NS","PIDILITIND.NS","DMART.NS","BRITANNIA.NS","SHREECEM.NS",
]

@st.cache_data(ttl=900)
def fetch_movers():
    results = []
    for ticker in NIFTY50:
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if len(hist) >= 2:
                curr = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                chg  = round(((curr - prev) / prev) * 100, 2)
                results.append({
                    "ticker": ticker,
                    "name":   ticker.replace(".NS", ""),
                    "price":  round(curr, 2),
                    "change": chg,
                })
        except Exception:
            continue
    results.sort(key=lambda x: x["change"], reverse=True)
    return results[:5], results[-5:][::-1]


# ── NEWS ──────────────────────────────────────────────────────────────────────
RSS_SOURCES = {
    "Moneycontrol":   "https://www.moneycontrol.com/rss/latestnews.xml",
    "Economic Times": "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
    "ET Markets":     "https://economictimes.indiatimes.com/markets/rss.cms",
    "Business Std":   "https://www.business-standard.com/rss/markets-106.rss",
}

def fetch_news(company_name: str, max_per_source: int = 3) -> list[dict]:
    results    = []
    name_lower = company_name.lower()
    words      = [w for w in name_lower.split() if len(w) > 3]
    search_terms = list({name_lower} | set(words)) if words else [name_lower]

    for source_name, url in RSS_SOURCES.items():
        try:
            feed  = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                if count >= max_per_source:
                    break
                title   = strip_html(entry.get("title", ""))
                summary = strip_html(
                    entry.get("summary", entry.get("description", "")))
                text = (title + " " + summary).lower()
                full_match = name_lower in text
                word_match = words and all(w in text for w in words)
                if full_match or word_match:
                    results.append({
                        "source":    source_name,
                        "title":     title,
                        "summary":   summary[:300],
                        "link":      entry.get("link", ""),
                        "published": entry.get("published", ""),
                    })
                    count += 1
        except Exception:
            continue
    return results


# ── MUTUAL FUND ADVISOR ───────────────────────────────────────────────────────
def get_mf_recommendations(
    inv_type: str, tenure_years: int, risk: str, amount: float,
) -> str:
    prompt = f"""You are a SEBI-registered mutual fund advisor for Indian investors.

Investment details:
- Type    : {inv_type} (SIP or Lump Sum)
- Amount  : ₹{amount:,.0f} {'per month' if inv_type == 'SIP' else 'one-time'}
- Tenure  : {tenure_years} years
- Risk    : {risk}

Recommend exactly 10 real Indian mutual funds available on Zerodha Coin / Groww.
For each fund provide:
1. Fund name (full official name)
2. Category (e.g. Large Cap, ELSS, Debt, Hybrid)
3. Why it suits this investor profile and tenure
4. Approximate historical 3Y/5Y CAGR
5. Minimum SIP / lump sum amount

Format as a numbered list with real fund names.
End with a 2-line summary of the overall strategy.
"""
    try:
        model    = genai.GenerativeModel(MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


# ── GEMINI STOCK ANALYSIS ─────────────────────────────────────────────────────
def get_gemini_analysis(
    company: str, data: dict,
    news_items: list[dict], buy_price: float = 0.0,
) -> str:
    news_block = (
        "\n".join(
            f"- [{i['source']}] {i['title']}"
            + (f": {i['summary'][:150]}" if i["summary"] else "")
            for i in news_items
        )
        if news_items
        else "No recent news found for this company in today's feeds."
    )
    holding_text = ""
    if buy_price > 0:
        pl     = round(data["price"] - buy_price, 2)
        pl_pct = round((pl / buy_price) * 100, 2)
        holding_text = f"""
The investor HOLDS this stock — bought at ₹{buy_price:.2f}.
Current P&L: ₹{pl:+.2f} ({pl_pct:+.2f}%)
Include a specific HOLD / BOOK PROFIT / AVERAGE DOWN call.
"""
    prompt = f"""You are a seasoned Indian equity analyst for a retail investor.

=== PRICE & TECHNICAL DATA ===
Company  : {company} ({data['ticker']})
Price    : ₹{data['price']}   |  Day change: ₹{data['change']} ({data['change_pct']}%)
52W High : ₹{data['high_52w']}  |  52W Low: ₹{data['low_52w']}
SMA 20   : ₹{data['sma20']}    |  SMA 50 : {data['sma50'] or 'N/A'}
Volume   : Today {data['last_vol']:,}  vs 20-day avg {data['avg_vol']:,}
{holding_text}
=== RECENT NEWS ===
{news_block}

Structure EXACTLY as:
**Verdict: BUY / HOLD / SKIP**
**What the price data says**
- (2–3 bullets with actual numbers)
**What the news says**
- (2–3 bullets referencing headlines above)
**Key risks**
- (2 bullets)
**Suggested action**
One sentence with price levels to watch.

Under 250 words. Simple language. No generic disclaimers.
"""
    try:
        model    = genai.GenerativeModel(MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


# ── EMAIL ─────────────────────────────────────────────────────────────────────
def send_wishlist_email(to_email: str, wishlist: list[dict]) -> bool:
    try:
        sender = st.secrets["EMAIL_ADDRESS"]
        pw     = st.secrets["EMAIL_APP_PASSWORD"]
        lines  = ["Your AI Stock Advisor Wishlist\n", "=" * 40]
        for item in wishlist:
            lines.append(
                f"\n{item['name']} ({item['ticker']})\n"
                f"  Price when added : ₹{item['price']}\n"
                f"  Added on         : {item['added']}"
            )
        lines.append("\n\n" + "=" * 40 + "\nNot SEBI-registered financial advice.")
        msg            = MIMEMultipart()
        msg["From"]    = sender
        msg["To"]      = to_email
        msg["Subject"] = "📋 Your Stock Wishlist — AI Stock Advisor"
        msg.attach(MIMEText("\n".join(lines), "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, pw)
            server.sendmail(sender, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Email error: {e}")
        return False


# ── SIDEBAR ── minimal now, just info ────────────────────────────────────────
with st.sidebar:
    st.header("📈 AI Stock Advisor")
    st.caption(f"Model: `{MODEL}`")
    st.divider()
    st.caption(
        "**How to use:**\n\n"
        "🏠 Market Overview — today's gainers & losers\n\n"
        "📊 Stock Analysis — search and analyse any stock\n\n"
        "💰 Mutual Funds — get fund recommendations\n\n"
        "⭐ Wishlist — save and track your stocks"
    )
    st.divider()
    st.caption("News: Moneycontrol, ET, Business Standard RSS.")
    st.caption("Educational use only — not SEBI-registered advice.")

    # Wishlist quick-view
    if st.session_state.wishlist:
        st.divider()
        st.markdown(f"⭐ **Wishlist ({len(st.session_state.wishlist)} stocks)**")
        for w in st.session_state.wishlist:
            st.caption(f"• {w['name']} — ₹{w['price']}")


# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.title("📈 AI Stock Advisor — India")
st.caption(
    f"Powered by `{MODEL}` + yfinance + RSS feeds. "
    "Educational use only — not SEBI-registered advice."
)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_market, tab_stocks, tab_mf, tab_wishlist = st.tabs([
    "🏠 Market Overview",
    "📊 Stock Analysis",
    "💰 Mutual Funds",
    f"⭐ Wishlist ({len(st.session_state.wishlist)})",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab_market:
    st.subheader("📈 Today's Top Gainers & Losers — Nifty 50")
    st.caption("Cached for 15 minutes. Click Refresh to force update.")

    if st.button("🔄 Refresh Market Data"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Fetching Nifty 50 movers..."):
        gainers, losers = fetch_movers()

    col_g, col_l = st.columns(2)

    with col_g:
        st.markdown("### 🟢 Top Gainers")
        for s in gainers:
            st.metric(
                label=s["name"],
                value=f"₹{s['price']}",
                delta=f"+{s['change']}%",
            )
            if st.button(
                f"Analyse {s['name']} ▶",
                key=f"g_{s['ticker']}",
                use_container_width=True,
            ):
                st.session_state.auto_ticker = s["ticker"]
                st.session_state.auto_name   = s["name"]
                st.rerun()

    with col_l:
        st.markdown("### 🔴 Top Losers")
        for s in losers:
            st.metric(
                label=s["name"],
                value=f"₹{s['price']}",
                delta=f"{s['change']}%",
            )
            if st.button(
                f"Analyse {s['name']} ▶",
                key=f"l_{s['ticker']}",
                use_container_width=True,
            ):
                st.session_state.auto_ticker = s["ticker"]
                st.session_state.auto_name   = s["name"]
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — STOCK ANALYSIS (input controls now live here)
# ════════════════════════════════════════════════════════════════════════════════
with tab_stocks:

    # ── INPUT FORM (replaces sidebar controls) ────────────────────────────────
    st.subheader("🔍 Search a Stock")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input(
            "Company Name or Ticker",
            placeholder="e.g. Wipro, HDFC Bank, RELIANCE.NS, 500570.BO",
            label_visibility="collapsed",
        )
    with col2:
        exchange = st.radio(
            "Exchange", ["NSE", "BSE"],
            horizontal=True,
            label_visibility="collapsed",
        )

    col3, col4 = st.columns([2, 2])
    with col3:
        is_holding = st.checkbox("I already hold this stock")
    with col4:
        buy_price = 0.0
        if is_holding:
            buy_price = st.number_input(
                "My buy price (₹)", min_value=0.01, step=0.5,
                label_visibility="visible",
            )

    analyze_btn = st.button(
        "Run Analysis ▶",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    # ── ANALYSIS LOGIC ────────────────────────────────────────────────────────
    do_analysis = False
    raw    = ""
    ticker = ""

    if st.session_state.auto_ticker:
        ticker      = st.session_state.auto_ticker
        raw         = st.session_state.auto_name
        st.session_state.auto_ticker = None
        st.session_state.auto_name   = None
        do_analysis = True

    elif analyze_btn and user_input.strip():
        raw = user_input.strip()
        if raw.upper().endswith((".NS", ".BO")):
            ticker = raw.upper()
        elif raw.strip().isdigit():
            ticker = f"{raw.strip()}.BO"
        else:
            with st.spinner(f"Looking up {exchange} ticker for '{raw}'..."):
                ticker = resolve_ticker(raw, exchange)
            if not ticker:
                st.error(f"Could not find ticker for **{raw}**.")
                st.info("Try pasting the ticker directly e.g. `WIPRO.NS`")
                st.stop()
        do_analysis = True

    elif analyze_btn and not user_input.strip():
        st.warning("Please enter a company name or ticker.")

    if do_analysis:
        st.info(f"Ticker resolved: `{ticker}`")
        with st.status("Fetching data...", expanded=True) as status:
            status.write(f"📊 Fetching price for `{ticker}`...")
            data = fetch_stock_data(ticker)
            if not data:
                st.error(
                    f"No data for `{ticker}`. "
                    "May be delisted or wrong symbol."
                )
                st.stop()
            status.write("📰 Scanning news feeds...")
            news_items = fetch_news(raw)
            status.write(f"   → {len(news_items)} headline(s) found")
            status.write("🤖 Running AI analysis...")
            analysis = get_gemini_analysis(raw, data, news_items, buy_price)
            status.update(label="Done!", state="complete", expanded=False)

        st.session_state.last_analysis = {
            "raw":       raw,
            "ticker":    ticker,
            "data":      data,
            "news_items": news_items,
            "analysis":  analysis,
            "buy_price": buy_price,
        }

    # ── DISPLAY RESULTS ───────────────────────────────────────────────────────
    if st.session_state.last_analysis:
        la = st.session_state.last_analysis
        d  = la["data"]

        st.subheader(f"📌 {la['raw'].title()}  ({la['ticker']})")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price",     f"₹{d['price']}",
                  f"₹{d['change']} ({d['change_pct']}%)")
        c2.metric("52W High",  f"₹{d['high_52w']}")
        c3.metric("52W Low",   f"₹{d['low_52w']}")
        c4.metric("SMA 20/50", f"₹{d['sma20']}",
                  delta=f"50d ₹{d['sma50']}" if d["sma50"] else "50d N/A",
                  delta_color="off")

        if la["buy_price"] > 0:
            pl     = round(d["price"] - la["buy_price"], 2)
            pl_pct = round((pl / la["buy_price"]) * 100, 2)
            st.subheader("💼 Your Position")
            p1, p2 = st.columns(2)
            p1.metric("Buy Price",      f"₹{la['buy_price']:.2f}")
            p2.metric("Unrealised P&L", f"₹{pl:+.2f}", f"{pl_pct:+.2f}%")

        st.subheader("3-Month Price Chart")
        st.line_chart(d["hist"][["Close"]].rename(
            columns={"Close": "Price (₹)"}))

        st.divider()
        already = any(
            w["ticker"] == la["ticker"] for w in st.session_state.wishlist)

        if already:
            st.success(f"✅ {la['raw'].title()} is already in your wishlist.")
            if st.button("❌ Remove from Wishlist"):
                st.session_state.wishlist = [
                    w for w in st.session_state.wishlist
                    if w["ticker"] != la["ticker"]
                ]
                save_wishlist(st.session_state.wishlist)
                st.rerun()
        else:
            if st.button(
                f"⭐ Add {la['raw'].title()} to Wishlist",
                type="primary",
                use_container_width=True,
            ):
                added = add_to_wishlist(
                    la["raw"].title(), la["ticker"], d["price"])
                if added:
                    st.success(
                        f"✅ {la['raw'].title()} added to wishlist! "
                        "View it in the ⭐ Wishlist tab."
                    )
                    st.rerun()

        st.divider()

        st.subheader(f"📰 Headlines ({len(la['news_items'])} found)")
        if la["news_items"]:
            for item in la["news_items"]:
                with st.expander(f"[{item['source']}] {item['title']}"):
                    if item["summary"]:
                        st.write(item["summary"])
                    if item["link"]:
                        st.markdown(f"[Read full article →]({item['link']})")
                    if item["published"]:
                        st.caption(f"Published: {item['published']}")
        else:
            st.info(
                "No matching headlines today. "
                "Analysis based on price data only."
            )

        st.subheader("🤖 AI Analysis")
        st.markdown(la["analysis"])
        st.divider()
        st.caption(
            f"Generated {datetime.now().strftime('%d %b %Y, %I:%M %p')} IST  |  "
            "Not SEBI-registered financial advice."
        )

    else:
        st.info(
            "Enter a company name above and click **Run Analysis ▶**\n\n"
            "Or click **Analyse ▶** on any stock in the 🏠 Market Overview tab."
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — MUTUAL FUNDS
# ════════════════════════════════════════════════════════════════════════════════
with tab_mf:
    st.subheader("💰 Mutual Fund Advisor")
    st.caption("Get 10 personalised fund recommendations based on your profile.")

    mf1, mf2 = st.columns(2)
    with mf1:
        inv_type = st.radio(
            "Investment type", ["SIP", "Lump Sum"], horizontal=True)
        amount   = st.number_input(
            "Amount (₹)" + (" per month" if inv_type == "SIP" else " one-time"),
            min_value=500.0, step=500.0, value=5000.0,
        )
    with mf2:
        risk = st.select_slider(
            "Risk appetite",
            options=["Very Low", "Low", "Moderate", "High", "Very High"],
            value="Moderate",
        )
        tenure = st.slider(
            "Investment tenure (years)", 1,
            30 if inv_type == "SIP" else 20, 5,
        )

    if st.button("Get Fund Recommendations 🔍", type="primary"):
        with st.spinner("Getting personalised fund recommendations..."):
            mf_result = get_mf_recommendations(inv_type, tenure, risk, amount)
        st.subheader("📋 Your Recommended Funds")
        st.markdown(mf_result)
        st.divider()
        st.caption(
            "AI-generated based on historical data. "
            "Past performance is not a guarantee of future returns. "
            "Not SEBI-registered financial advice."
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — WISHLIST
# ════════════════════════════════════════════════════════════════════════════════
with tab_wishlist:
    st.subheader("⭐ Your Wishlist")

    if not st.session_state.wishlist:
        st.info(
            "Your wishlist is empty.\n\n"
            "**How to add stocks:**\n"
            "1. Go to 📊 Stock Analysis tab\n"
            "2. Search and analyse any stock\n"
            "3. Click the ⭐ Add to Wishlist button\n\n"
            "Or click **Analyse ▶** on any stock in the 🏠 Market Overview tab."
        )
    else:
        st.caption(f"{len(st.session_state.wishlist)} stock(s) saved")
        st.divider()

        for i, item in enumerate(st.session_state.wishlist):
            col_a, col_b, col_c, col_d = st.columns([3, 2, 2, 1])
            with col_a:
                st.markdown(f"**{item['name']}**")
                st.caption(f"`{item['ticker']}` · Added {item['added']}")
            with col_b:
                st.metric("Price when added", f"₹{item['price']}")
            with col_c:
                try:
                    live_hist = yf.Ticker(item["ticker"]).history(period="1d")
                    if not live_hist.empty:
                        live      = round(float(live_hist["Close"].iloc[-1]), 2)
                        delta     = round(live - item["price"], 2)
                        delta_pct = round((delta / item["price"]) * 100, 2)
                        st.metric(
                            "Live Price",
                            f"₹{live}",
                            f"₹{delta:+.2f} ({delta_pct:+.2f}%) since added",
                        )
                except Exception:
                    st.caption("Live price unavailable")
            with col_d:
                if st.button("🗑️", key=f"del_{i}", help="Remove from wishlist"):
                    st.session_state.wishlist.pop(i)
                    save_wishlist(st.session_state.wishlist)
                    st.rerun()

            st.divider()

        if st.button("🗑️ Clear Entire Wishlist", type="secondary"):
            st.session_state.wishlist = []
            save_wishlist(st.session_state.wishlist)
            st.rerun()

        st.subheader("📧 Email Your Wishlist")
        st.caption(
            "Add EMAIL_ADDRESS and EMAIL_APP_PASSWORD to Streamlit Secrets. "
            "Use a Gmail App Password from "
            "myaccount.google.com → Security → App Passwords."
        )
        to_email  = st.text_input("Your email address", placeholder="you@gmail.com")
        email_btn = st.button("Send Wishlist to Email 📨", type="primary")

        if email_btn:
            if not to_email or "@" not in to_email:
                st.warning("Please enter a valid email address.")
            elif "EMAIL_ADDRESS" not in st.secrets:
                st.error(
                    "EMAIL_ADDRESS and EMAIL_APP_PASSWORD not in Secrets. "
                    "Add them in Streamlit Cloud → Settings → Secrets."
                )
            else:
                with st.spinner("Sending..."):
                    ok = send_wishlist_email(to_email, st.session_state.wishlist)
                if ok:
                    st.success(f"✅ Wishlist sent to {to_email}!")
