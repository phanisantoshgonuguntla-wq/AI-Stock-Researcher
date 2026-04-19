import streamlit as st
import yfinance as yf
import google.generativeai as genai
import feedparser
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Advisor (India)", layout="wide", page_icon="📈")

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("🔑 Add GOOGLE_API_KEY to Streamlit Secrets (Settings → Secrets).")
    st.stop()


# ── AUTO-DETECT BEST AVAILABLE MODEL ─────────────────────────────────────────
def get_best_model() -> str:
    preferred = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
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


# ── TICKER RESOLVER ───────────────────────────────────────────────────────────
def clean_ticker(raw: str, suffix: str) -> str | None:
    """Clean and validate a raw ticker string returned by Gemini."""
    result = raw.strip().upper()
    result = result.split("\n")[0].strip()       # first line only
    result = result.replace("'", "").replace('"', "")
    result = result.replace(" ", "")             # remove any spaces
    while ".." in result:
        result = result.replace("..", ".")
    result = result.rstrip(".,;:")

    if not result or result == "UNKNOWN":
        return None

    # Must be at least 2 characters before the suffix
    base = result.replace(".NS", "").replace(".BO", "")
    if len(base) < 2:
        return None

    # Add suffix if missing
    if not (result.endswith(".NS") or result.endswith(".BO")):
        result = result.split()[0] + suffix

    return result


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

Correct examples of complete ticker symbols:
WIPRO.NS
RELIANCE.NS
HDFCBANK.NS
TCS.NS
INFY.NS
SBIN.NS
ZOMATO.NS
TATAMOTORS.NS
BAJFINANCE.NS
ICICIBANK.NS
500570.BO
532540.BO

If truly unknown, reply: UNKNOWN

Complete ticker symbol for {company_name}:"""

    try:
        model = genai.GenerativeModel(MODEL)

        # ── First attempt ─────────────────────────────────────────────────────
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=200,   # raised significantly — prevents truncation
                temperature=0.0,
            ),
        )
        result = clean_ticker(response.text, suffix)

        # ── Validate: if result looks truncated, retry with explicit length hint
        if result:
            base = result.replace(".NS", "").replace(".BO", "")
            if len(base) < 3:
                # Looks truncated — retry with stronger instruction
                retry_prompt = f"""The NSE ticker symbol for {company_name} is a string of letters followed by .NS
Write the FULL ticker symbol without cutting it short.
For example Wipro's full ticker is WIPRO.NS not WIP.NS
Write the complete ticker for {company_name} on {exchange}:"""

                retry_response = model.generate_content(
                    retry_prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=200,
                        temperature=0.0,
                    ),
                )
                result = clean_ticker(retry_response.text, suffix)

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
        sma50    = round(float(hist["Close"].tail(50).mean()), 2) if len(hist) >= 50 else None
        change     = round(curr - prev, 2)
        change_pct = round((change / prev) * 100, 2)
        avg_vol    = int(hist["Volume"].tail(20).mean())
        last_vol   = int(hist["Volume"].iloc[-1])

        return {
            "ticker":     ticker,
            "price":      curr,
            "change":     change,
            "change_pct": change_pct,
            "high_52w":   high_52w,
            "low_52w":    low_52w,
            "sma20":      sma20,
            "sma50":      sma50,
            "avg_vol":    avg_vol,
            "last_vol":   last_vol,
            "hist":       hist,
        }
    except Exception as e:
        st.error(f"Price fetch error: {e}")
        return None


# ── NEWS ──────────────────────────────────────────────────────────────────────
RSS_SOURCES = {
    "Moneycontrol":   "https://www.moneycontrol.com/rss/latestnews.xml",
    "Economic Times": "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
    "ET Markets":     "https://economictimes.indiatimes.com/markets/rss.cms",
    "Business Std":   "https://www.business-standard.com/rss/markets-106.rss",
}

def fetch_news(company_name: str, max_per_source: int = 3) -> list[dict]:
    results      = []
    search_terms = [w.lower() for w in company_name.split() if len(w) > 3]

    # If the entire name is short (e.g. "SBI", "ITC"), use it as-is
    if not search_terms:
        search_terms = [company_name.lower()]

    for source_name, url in RSS_SOURCES.items():
        try:
            feed  = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                if count >= max_per_source:
                    break
                title   = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                text    = (title + " " + summary).lower()
                if any(term in text for term in search_terms):
                    results.append({
                        "source":    source_name,
                        "title":     title,
                        "summary":   summary[:300] if summary else "",
                        "link":      entry.get("link", ""),
                        "published": entry.get("published", ""),
                    })
                    count += 1
        except Exception:
            continue

    return results


# ── GEMINI ANALYSIS ───────────────────────────────────────────────────────────
def get_gemini_analysis(
    company: str,
    data: dict,
    news_items: list[dict],
    buy_price: float = 0.0,
) -> str:

    news_block = (
        "\n".join(
            f"- [{item['source']}] {item['title']}"
            + (f": {item['summary'][:150]}" if item["summary"] else "")
            for item in news_items
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
Include a specific HOLD / BOOK PROFIT / AVERAGE DOWN call for them.
"""

    prompt = f"""You are a seasoned Indian equity analyst writing for a retail investor.
Analyse the data below and give a clear, actionable recommendation.

=== PRICE & TECHNICAL DATA ===
Company  : {company} ({data['ticker']})
Price    : ₹{data['price']}   |  Day change: ₹{data['change']} ({data['change_pct']}%)
52W High : ₹{data['high_52w']}  |  52W Low: ₹{data['low_52w']}
SMA 20   : ₹{data['sma20']}    |  SMA 50 : {data['sma50'] or 'N/A'}
Volume   : Today {data['last_vol']:,}  vs 20-day avg {data['avg_vol']:,}
{holding_text}
=== RECENT NEWS (Moneycontrol / Economic Times / Business Standard) ===
{news_block}

=== INSTRUCTIONS ===
Structure your response EXACTLY like this (use these bold headers):

**Verdict: BUY / HOLD / SKIP**

**What the price data says**
- (2–3 bullet points using the numbers above)

**What the news says**
- (2–3 bullet points referencing the actual headlines above; if no news, say so)

**Key risks**
- (2 bullet points)

**Suggested action**
One sentence: what to do and at what price level to watch.

Keep it under 250 words. Use simple language. Be specific with rupee values.
Do not add generic disclaimers.
"""

    try:
        model    = genai.GenerativeModel(MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Analyse a Stock")

    exchange = st.radio("Exchange", ["NSE", "BSE"], horizontal=True)
    st.caption("NSE recommended — better data quality on yfinance.")

    user_input = st.text_input(
        "Company Name or Ticker",
        placeholder="e.g. Wipro, HDFC Bank, RELIANCE.NS, 500570.BO",
    )

    st.divider()
    is_holding = st.checkbox("I already hold this stock")
    buy_price  = 0.0
    if is_holding:
        buy_price = st.number_input("My buy price (₹)", min_value=0.01, step=0.5)

    analyze_btn = st.button("Run Analysis ▶", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "News from Moneycontrol, Economic Times & "
        "Business Standard RSS — free, no API key.\n\n"
        "Tip: If lookup fails, paste the ticker directly "
        "e.g. `WIPRO.NS` or `507685.BO`."
    )
    st.caption(f"Model in use: `{MODEL}`")


# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.title("📈 AI Stock Advisor — India")
st.caption(
    f"Powered by `{MODEL}` + yfinance + MC / ET / BS RSS.  "
    "For educational use only — not SEBI-registered advice."
)


# ── MAIN FLOW ─────────────────────────────────────────────────────────────────
if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter a company name or ticker.")
        st.stop()

    raw = user_input.strip()

    # ── Step 1: Resolve ticker ────────────────────────────────────────────────
    if raw.upper().endswith((".NS", ".BO")):
        ticker = raw.upper()
        st.info(f"Using ticker directly: `{ticker}`")
    elif raw.strip().isdigit():
        ticker = f"{raw.strip()}.BO"
        st.info(f"Using BSE code directly: `{ticker}`")
    else:
        with st.spinner(f"Looking up {exchange} ticker for '{raw}'..."):
            ticker = resolve_ticker(raw, exchange)

        if not ticker:
            st.error(f"Could not find a {exchange} ticker for **{raw}**.")
            st.info(
                "Try being more specific (e.g. 'Tata Consultancy Services' "
                "instead of 'TCS'), or paste the ticker directly "
                "(e.g. `TCS.NS` or `532540.BO`)."
            )
            st.stop()
        else:
            st.info(f"Ticker resolved: `{ticker}`")

    # ── Step 2: Fetch price + news + analysis ────────────────────────────────
    with st.status("Fetching data...", expanded=True) as status:

        status.write(f"📊 Getting live price for `{ticker}` from yfinance...")
        data = fetch_stock_data(ticker)

        if not data:
            st.error(
                f"No price data returned for `{ticker}`. "
                "It may be delisted, or the symbol may be wrong."
            )
            st.stop()

        status.write("📰 Scanning Moneycontrol, ET, Business Standard RSS feeds...")
        news_items = fetch_news(raw)
        status.write(f"   → {len(news_items)} relevant headline(s) found")

        status.write(f"🤖 Sending everything to `{MODEL}` for analysis...")
        analysis = get_gemini_analysis(raw, data, news_items, buy_price)

        status.update(label="Analysis complete!", state="complete", expanded=False)

    # ── METRICS ───────────────────────────────────────────────────────────────
    st.subheader(f"📌 {raw.title()}  ({ticker})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price",     f"₹{data['price']}",
              f"₹{data['change']} ({data['change_pct']}%)")
    c2.metric("52W High",  f"₹{data['high_52w']}")
    c3.metric("52W Low",   f"₹{data['low_52w']}")
    c4.metric("SMA 20/50", f"₹{data['sma20']}",
              delta=f"50d ₹{data['sma50']}" if data["sma50"] else "50d N/A",
              delta_color="off")

    # ── P&L FOR HOLDERS ───────────────────────────────────────────────────────
    if buy_price > 0:
        pl     = round(data["price"] - buy_price, 2)
        pl_pct = round((pl / buy_price) * 100, 2)
        st.subheader("💼 Your Position")
        p1, p2 = st.columns(2)
        p1.metric("Buy Price",      f"₹{buy_price:.2f}")
        p2.metric("Unrealised P&L", f"₹{pl:+.2f}", f"{pl_pct:+.2f}%")

    # ── CHART ─────────────────────────────────────────────────────────────────
    st.subheader("3-Month Price Chart")
    st.line_chart(data["hist"][["Close"]].rename(columns={"Close": "Price (₹)"}))

    # ── NEWS ──────────────────────────────────────────────────────────────────
    st.subheader(f"📰 Recent Headlines ({len(news_items)} found)")
    if news_items:
        for item in news_items:
            with st.expander(f"[{item['source']}] {item['title']}"):
                if item["summary"]:
                    st.write(item["summary"])
                if item["link"]:
                    st.markdown(f"[Read full article →]({item['link']})")
                if item["published"]:
                    st.caption(f"Published: {item['published']}")
    else:
        st.info(
            "No headlines matched this company in today's feeds. "
            "AI analysis is based on price data only."
        )

    # ── AI ANALYSIS ───────────────────────────────────────────────────────────
    st.subheader("🤖 AI Analysis")
    st.markdown(analysis)

    st.divider()
    st.caption(
        f"Generated {datetime.now().strftime('%d %b %Y, %I:%M %p')} IST  |  "
        "Not SEBI-registered financial advice."
    )
