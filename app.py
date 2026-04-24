import streamlit as st
import yfinance as yf
from google import genai
from google.genai import types as genai_types
import feedparser
import smtplib
import re
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Advisor (India)", layout="wide", page_icon="📈")

try:
    gemini_client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
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
if "chat_history"  not in st.session_state:
    st.session_state.chat_history  = []


# ── AUTO-DETECT BEST MODEL ────────────────────────────────────────────────────
def get_best_model() -> str:
    preferred = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    try:
        available = [m.name for m in gemini_client.models.list()]
        for model in preferred:
            if any(model in a for a in available):
                return model
    except Exception:
        pass
    return "gemini-2.0-flash"

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


def clean_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a yfinance DataFrame regardless of version.

    yfinance ≥ 0.2.50 returns MultiIndex columns like:
        ('Close', 'RELIANCE.NS'), ('Open', 'RELIANCE.NS'), …
    Older versions return flat columns: 'Close', 'Open', …

    After this function every column is a plain string and duplicates
    are removed, so df['Close'] always yields a plain Series.
    """
    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        # Keep only the price-field level (level 0) and drop the ticker level
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # After flattening we may still have duplicate column names
    # (e.g. two 'Close' columns if two tickers were downloaded).
    # Keep only the first occurrence.
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Extract *col* from *df* and guarantee a 1-D pd.Series is returned.

    Handles every yfinance edge-case:
    - Column is already a Series → return as-is
    - Column is a 1-column DataFrame (can happen after MultiIndex flatten) → take iloc[:,0]
    - squeeze() on a single-element Series would give a scalar → guard against that
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available: {df.columns.tolist()}")

    s = df[col]

    # If it's still a DataFrame (duplicate columns survived), take the first
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    # Ensure it really is a Series (not a scalar from a 1-row df)
    if not isinstance(s, pd.Series):
        s = pd.Series([s], index=df.index)

    return s


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


# ── TECHNICAL INDICATORS ──────────────────────────────────────────────────────
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta    = prices.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast    = prices.ewm(span=fast,   adjust=False).mean()
    ema_slow    = prices.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger(prices: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma   = prices.rolling(period).mean()
    std   = prices.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


# ── CANDLESTICK + INDICATORS CHART ───────────────────────────────────────────
def build_chart(hist: pd.DataFrame, ticker: str) -> go.Figure:
    def _flat(col):
        s = hist[col]
        if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
        return pd.Series(s.to_numpy().flatten().astype(float), index=hist.index[:len(s)])
    close  = _flat("Close")
    open_  = _flat("Open")
    high   = _flat("High")
    low    = _flat("Low")
    volume = _flat("Volume")

    rsi                               = calculate_rsi(close)
    macd_line, signal_line, histogram = calculate_macd(close)
    bb_upper, bb_mid, bb_lower        = calculate_bollinger(close)
    sma50 = close.rolling(50).mean()

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.18, 0.17],
        vertical_spacing=0.03,
        subplot_titles=(
            f"{ticker} — Price + Bollinger Bands",
            "Volume",
            "RSI (14)",
            "MACD (12/26/9)",
        ),
    )

    # Panel 1: Candlestick
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=open_,
        high=high,
        low=low,
        close=close,
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index, y=bb_upper,
        line=dict(color="rgba(100,100,255,0.4)", width=1, dash="dot"),
        name="BB Upper", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=bb_lower,
        line=dict(color="rgba(100,100,255,0.4)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(100,100,255,0.05)",
        name="BB Lower", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=bb_mid,
        line=dict(color="rgba(100,100,255,0.6)", width=1),
        name="BB Mid", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=close.rolling(20).mean(),
        line=dict(color="#f39c12", width=1.5),
        name="SMA 20",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=sma50,
        line=dict(color="#8e44ad", width=1.5),
        name="SMA 50",
    ), row=1, col=1)

    # Panel 2: Volume — colour by up/down candle
    colors = [
        "#26a69a" if float(close.iloc[i]) >= float(open_.iloc[i])
        else "#ef5350"
        for i in range(len(close))
    ]
    fig.add_trace(go.Bar(
        x=hist.index, y=volume,
        marker_color=colors,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    # Panel 3: RSI
    fig.add_trace(go.Scatter(
        x=hist.index, y=rsi,
        line=dict(color="#e67e22", width=1.5),
        name="RSI",
    ), row=3, col=1)
    for level, color in [(70, "rgba(239,83,80,0.5)"), (30, "rgba(38,166,154,0.5)")]:
        fig.add_hline(
            y=level, line_dash="dot",
            line_color=color, line_width=1,
            row=3, col=1,
        )

    # Panel 4: MACD
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in histogram]
    fig.add_trace(go.Bar(
        x=hist.index, y=histogram,
        marker_color=hist_colors,
        name="MACD Hist", showlegend=False,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=macd_line,
        line=dict(color="#3498db", width=1.5),
        name="MACD",
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=signal_line,
        line=dict(color="#e74c3c", width=1.5),
        name="Signal",
    ), row=4, col=1)

    fig.update_layout(
        height=750,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)", showgrid=True)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", showgrid=True)
    return fig


# ── TECHNICAL SUMMARY FOR GEMINI ─────────────────────────────────────────────
def get_technical_summary(hist: pd.DataFrame) -> dict:
    raw_c = hist["Close"]
    if isinstance(raw_c, pd.DataFrame): raw_c = raw_c.iloc[:, 0]
    close = pd.Series(raw_c.to_numpy().flatten().astype(float), index=hist.index)

    rsi                               = calculate_rsi(close)
    macd_line, signal_line, histogram = calculate_macd(close)
    bb_upper, bb_mid, bb_lower        = calculate_bollinger(close)

    last_rsi    = round(float(rsi.iloc[-1]), 2)
    last_macd   = round(float(macd_line.iloc[-1]), 4)
    last_signal = round(float(signal_line.iloc[-1]), 4)
    last_hist   = round(float(histogram.iloc[-1]), 4)
    last_close  = round(float(close.iloc[-1]), 2)
    last_bb_up  = round(float(bb_upper.iloc[-1]), 2)
    last_bb_low = round(float(bb_lower.iloc[-1]), 2)

    if last_rsi > 70:
        rsi_signal = "Overbought — potential reversal or pullback risk"
    elif last_rsi < 30:
        rsi_signal = "Oversold — potential bounce or buying opportunity"
    else:
        rsi_signal = "Neutral zone"

    if last_macd > last_signal and last_hist > 0:
        macd_signal = "Bullish — MACD above signal line, positive histogram"
    elif last_macd < last_signal and last_hist < 0:
        macd_signal = "Bearish — MACD below signal line, negative histogram"
    else:
        macd_signal = "Crossover zone — momentum changing"

    if last_close > last_bb_up:
        bb_signal = "Price above upper band — overbought / strong breakout"
    elif last_close < last_bb_low:
        bb_signal = "Price below lower band — oversold / strong breakdown"
    else:
        pct = round(
            (last_close - last_bb_low) / (last_bb_up - last_bb_low) * 100, 1)
        bb_signal = f"Price within bands at {pct}% of band width"

    return {
        "rsi":         last_rsi,
        "rsi_signal":  rsi_signal,
        "macd":        last_macd,
        "macd_signal": macd_signal,
        "bb_upper":    last_bb_up,
        "bb_lower":    last_bb_low,
        "bb_signal":   bb_signal,
    }


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
        response = gemini_client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=200, temperature=0.0),
        )
        result = clean_ticker(response.text, suffix)
        if result:
            base = result.replace(".NS", "").replace(".BO", "")
            if len(base) < 3:
                retry = gemini_client.models.generate_content(
                    model=MODEL,
                    contents=(
                        f"Write the FULL NSE ticker for {company_name} ending in "
                        f"{suffix}. Example: Wipro = WIPRO.NS. Just the ticker:"
                    ),
                    config=genai_types.GenerateContentConfig(
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
        asset    = yf.Ticker(ticker)
        raw_hist = asset.history(period="6mo")

        # DEBUG block - shows raw yfinance output in sidebar
        with st.sidebar.expander("yfinance debug", expanded=False):
            st.write("Raw columns:", raw_hist.columns.tolist())
            st.write("Column type:", type(raw_hist.columns).__name__)
            st.write("Shape:", raw_hist.shape)
            if not raw_hist.empty:
                st.write("Last row (raw):", raw_hist.tail(1))

        hist = clean_yf_df(raw_hist)
        if hist.empty:
            return None

        def extract(col):
            s = hist[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            arr = s.to_numpy().flatten().astype(float)
            return pd.Series(arr, index=hist.index[:len(arr)])

        close  = extract("Close")
        open_  = extract("Open")
        high   = extract("High")
        low    = extract("Low")
        volume = extract("Volume")

        curr       = round(float(close.iloc[-1]), 2)
        prev       = round(float(close.iloc[-2]), 2)
        high_52w   = round(float(high.max()), 2)
        low_52w    = round(float(low.min()), 2)
        sma20      = round(float(close.tail(20).mean()), 2)
        sma50      = (round(float(close.tail(50).mean()), 2)
                      if len(hist) >= 50 else None)
        change     = round(curr - prev, 2)
        change_pct = round((change / prev) * 100, 2)
        avg_vol    = int(volume.tail(20).mean())
        last_vol   = int(volume.iloc[-1])

        hist = hist.copy()
        hist["Close"]  = close.values
        hist["Open"]   = open_.values
        hist["High"]   = high.values
        hist["Low"]    = low.values
        hist["Volume"] = volume.values

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
        import traceback
        st.error(f"Price fetch error: {e}")
        st.code(traceback.format_exc())
        return None


# ── HISTORICAL P&L SIMULATOR ──────────────────────────────────────────────────
def simulate_pnl(ticker: str, amount: float, inv_date: str) -> dict | None:
    try:
        start = datetime.strptime(inv_date, "%Y-%m-%d")
        end   = datetime.now()

        # Use Ticker().history() to avoid yf.download() MultiIndex issues
        raw_hist = clean_yf_df(
            yf.Ticker(ticker).history(start=start, end=end))
        if raw_hist.empty or len(raw_hist) < 2:
            return None

        raw_close = raw_hist["Close"]
        if isinstance(raw_close, pd.DataFrame):
            raw_close = raw_close.iloc[:, 0]
        close = pd.Series(raw_close.to_numpy().flatten().astype(float), index=raw_hist.index)
        buy_price  = float(close.iloc[0])
        sell_price = float(close.iloc[-1])
        shares     = amount / buy_price
        curr_value = shares * sell_price
        pnl        = curr_value - amount
        pnl_pct    = (pnl / amount) * 100
        days       = (end - start).days
        years      = days / 365.25
        cagr       = ((curr_value / amount) ** (1 / years) - 1) * 100 \
                     if years > 0 else 0

        # Nifty 50 comparison
        nifty_return = 0.0
        nifty_norm   = None
        raw_nifty = clean_yf_df(
            yf.Ticker("^NSEI").history(start=start, end=end))
        if not raw_nifty.empty:
            raw_nc = raw_nifty["Close"]
            if isinstance(raw_nc, pd.DataFrame):
                raw_nc = raw_nc.iloc[:, 0]
            nifty_close = pd.Series(raw_nc.to_numpy().flatten().astype(float), index=raw_nifty.index)
            n_start      = float(nifty_close.iloc[0])
            n_end        = float(nifty_close.iloc[-1])
            nifty_return = ((n_end - n_start) / n_start) * 100
            nifty_norm   = (nifty_close / nifty_close.iloc[0]) * amount

        # Normalised growth chart
        stock_norm = (close / close.iloc[0]) * amount

        return {
            "buy_price":    round(buy_price, 2),
            "sell_price":   round(sell_price, 2),
            "shares":       round(shares, 4),
            "invested":     round(amount, 2),
            "curr_value":   round(curr_value, 2),
            "pnl":          round(pnl, 2),
            "pnl_pct":      round(pnl_pct, 2),
            "cagr":         round(cagr, 2),
            "nifty_return": round(nifty_return, 2),
            "stock_norm":   stock_norm,
            "nifty_norm":   nifty_norm,
            "days":         days,
        }

    except Exception as e:
        st.error(f"Simulator error: {e}")
        return None


# ── TOP GAINERS / LOSERS ──────────────────────────────────────────────────────
NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","HCLTECH.NS",
    "WIPRO.NS","SUNPHARMA.NS","TITAN.NS","BAJFINANCE.NS","NESTLEIND.NS",
    "ULTRACEMCO.NS","TECHM.NS","POWERGRID.NS","NTPC.NS","ONGC.NS",
    "M&M.NS","TATASTEEL.NS","JSWSTEEL.NS","ADANIENT.NS","ADANIPORTS.NS",
    "COALINDIA.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","EICHERMOT.NS","DIVISLAB.NS",
    "DRREDDY.NS","CIPLA.NS","APOLLOHOSP.NS","SBILIFE.NS","HDFCLIFE.NS",
    "HINDALCO.NS","BPCL.NS","IOC.NS","GRASIM.NS","TATACONSUM.NS",
    "PIDILITIND.NS","DMART.NS","BRITANNIA.NS","SHREECEM.NS","BAJAJFINSV.NS",
]

@st.cache_data(ttl=900)
def fetch_movers():
    results = []
    for ticker in NIFTY50:
        try:
            hist = clean_yf_df(yf.Ticker(ticker).history(period="2d"))
            if hist.empty or len(hist) < 2:
                continue
            raw_close = hist["Close"]
            if isinstance(raw_close, pd.DataFrame):
                raw_close = raw_close.iloc[:, 0]
            close = pd.Series(raw_close.to_numpy().flatten().astype(float), index=hist.index)
            curr  = float(close.iloc[-1])
            prev  = float(close.iloc[-2])
            chg   = round(((curr - prev) / prev) * 100, 2)
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
        response = gemini_client.models.generate_content(
            model=MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


# ── GEMINI STOCK ANALYSIS ─────────────────────────────────────────────────────
def get_gemini_analysis(
    company: str, data: dict, tech: dict,
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

=== TECHNICAL INDICATORS ===
RSI (14) : {tech['rsi']} — {tech['rsi_signal']}
MACD     : {tech['macd']} — {tech['macd_signal']}
Bollinger: Upper ₹{tech['bb_upper']} / Lower ₹{tech['bb_lower']} — {tech['bb_signal']}
{holding_text}
=== RECENT NEWS ===
{news_block}

Structure EXACTLY as:
**Verdict: BUY / HOLD / SKIP**
**What the price & indicators say**
- (3–4 bullets referencing RSI, MACD, Bollinger, SMA with actual numbers)
**What the news says**
- (2–3 bullets referencing actual headlines above)
**Key risks**
- (2 bullets)
**Suggested action**
One sentence with specific price levels to watch.

Under 300 words. Simple language. No generic disclaimers.
"""
    try:
        response = gemini_client.models.generate_content(
            model=MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


# ── ASK AI ANYTHING ───────────────────────────────────────────────────────────
def ask_ai(question: str, context: dict) -> str:
    data = context["data"]
    tech = context.get("tech", {})
    prompt = f"""You are an Indian stock market expert assistant.
The user is currently viewing an analysis of {context['raw'].title()} ({context['ticker']}).

Current context:
- Price    : ₹{data['price']} ({data['change_pct']}% today)
- RSI      : {tech.get('rsi', 'N/A')} — {tech.get('rsi_signal', '')}
- MACD     : {tech.get('macd_signal', 'N/A')}
- Bollinger: {tech.get('bb_signal', 'N/A')}
- 52W High : ₹{data['high_52w']} | 52W Low: ₹{data['low_52w']}
- SMA 20   : ₹{data['sma20']} | SMA 50: {data['sma50'] or 'N/A'}

Previous AI verdict: {context.get('analysis', '')[:300]}

User question: {question}

Answer clearly and concisely in under 150 words.
Use simple language suitable for a retail investor.
If the question is not related to this stock or markets, politely redirect.
"""
    try:
        response = gemini_client.models.generate_content(
            model=MODEL, contents=prompt)
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
        lines.append(
            "\n\n" + "=" * 40 + "\nNot SEBI-registered financial advice.")
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


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📈 AI Stock Advisor")
    st.caption(f"Model: `{MODEL}`")
    st.divider()
    st.caption(
        "**How to use:**\n\n"
        "🏠 Market Overview — today's gainers & losers\n\n"
        "📊 Stock Analysis — search and analyse any stock\n\n"
        "🧮 P&L Simulator — what-if historical returns\n\n"
        "💰 Mutual Funds — get fund recommendations\n\n"
        "⭐ Wishlist — save and track your stocks"
    )
    st.divider()
    st.caption("News: Moneycontrol, ET, Business Standard RSS.")
    st.caption("Educational use only — not SEBI-registered advice.")

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
tab_market, tab_stocks, tab_simulator, tab_mf, tab_wishlist = st.tabs([
    "🏠 Market Overview",
    "📊 Stock Analysis",
    "🧮 P&L Simulator",
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
            st.metric(label=s["name"], value=f"₹{s['price']}",
                      delta=f"+{s['change']}%")
            if st.button(f"Analyse {s['name']} ▶",
                         key=f"g_{s['ticker']}", use_container_width=True):
                st.session_state.auto_ticker = s["ticker"]
                st.session_state.auto_name   = s["name"]
                st.rerun()

    with col_l:
        st.markdown("### 🔴 Top Losers")
        for s in losers:
            st.metric(label=s["name"], value=f"₹{s['price']}",
                      delta=f"{s['change']}%")
            if st.button(f"Analyse {s['name']} ▶",
                         key=f"l_{s['ticker']}", use_container_width=True):
                st.session_state.auto_ticker = s["ticker"]
                st.session_state.auto_name   = s["name"]
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — STOCK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tab_stocks:

    st.subheader("🔍 Search a Stock")
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input(
            "Company Name or Ticker",
            placeholder="e.g. Wipro, HDFC Bank, RELIANCE.NS",
            label_visibility="collapsed",
        )
    with col2:
        exchange = st.radio("Exchange", ["NSE", "BSE"],
                            horizontal=True, label_visibility="collapsed")

    col3, col4 = st.columns([2, 2])
    with col3:
        is_holding = st.checkbox("I already hold this stock")
    with col4:
        buy_price = 0.0
        if is_holding:
            buy_price = st.number_input("My buy price (₹)",
                                        min_value=0.01, step=0.5)

    analyze_btn = st.button("Run Analysis ▶", type="primary",
                            use_container_width=True)
    st.divider()

    do_analysis = False
    raw = ticker = ""

    if st.session_state.auto_ticker:
        ticker  = st.session_state.auto_ticker
        raw     = st.session_state.auto_name
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
            status.write(f"📊 Fetching price + indicators for `{ticker}`...")
            data = fetch_stock_data(ticker)
            if not data:
                st.error(
                    f"No data for `{ticker}`. May be delisted or wrong symbol.")
                st.stop()

            status.write("📐 Calculating RSI, MACD, Bollinger Bands...")
            tech = get_technical_summary(data["hist"])

            status.write("📰 Scanning news feeds...")
            news_items = fetch_news(raw)
            status.write(f"   → {len(news_items)} headline(s) found")

            status.write("🤖 Running AI analysis with all indicators...")
            analysis = get_gemini_analysis(
                raw, data, tech, news_items, buy_price)
            status.update(label="Done!", state="complete", expanded=False)

        st.session_state.last_analysis = {
            "raw":        raw,
            "ticker":     ticker,
            "data":       data,
            "tech":       tech,
            "news_items": news_items,
            "analysis":   analysis,
            "buy_price":  buy_price,
        }
        st.session_state.chat_history = []

    if st.session_state.last_analysis:
        la   = st.session_state.last_analysis
        d    = la["data"]
        tech = la["tech"]

        st.subheader(f"📌 {la['raw'].title()}  ({la['ticker']})")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price",     f"₹{d['price']}",
                  f"₹{d['change']} ({d['change_pct']}%)")
        c2.metric("52W High",  f"₹{d['high_52w']}")
        c3.metric("52W Low",   f"₹{d['low_52w']}")
        c4.metric("SMA 20/50", f"₹{d['sma20']}",
                  delta=f"50d ₹{d['sma50']}" if d["sma50"] else "50d N/A",
                  delta_color="off")

        i1, i2, i3 = st.columns(3)
        rsi_color = (
            "inverse" if tech["rsi"] > 70
            else "normal" if tech["rsi"] < 30
            else "off"
        )
        i1.metric("RSI (14)", f"{tech['rsi']}",
                  tech["rsi_signal"].split("—")[0].strip(),
                  delta_color=rsi_color)
        macd_delta_color = (
            "normal"  if "Bullish" in tech["macd_signal"]
            else "inverse" if "Bearish" in tech["macd_signal"]
            else "off"
        )
        i2.metric(
            "MACD Signal",
            "Bullish" if "Bullish" in tech["macd_signal"]
            else "Bearish" if "Bearish" in tech["macd_signal"]
            else "Neutral",
            delta_color=macd_delta_color,
        )
        i3.metric("Bollinger",
                  f"₹{tech['bb_lower']} – ₹{tech['bb_upper']}",
                  tech["bb_signal"][:30],
                  delta_color="off")

        if la["buy_price"] > 0:
            pl     = round(d["price"] - la["buy_price"], 2)
            pl_pct = round((pl / la["buy_price"]) * 100, 2)
            st.subheader("💼 Your Position")
            p1, p2 = st.columns(2)
            p1.metric("Buy Price",      f"₹{la['buy_price']:.2f}")
            p2.metric("Unrealised P&L", f"₹{pl:+.2f}", f"{pl_pct:+.2f}%")

        st.subheader("📊 Interactive Chart")
        st.caption(
            "Candlestick + Bollinger Bands + SMA 20/50 | "
            "Volume | RSI (14) | MACD (12/26/9) — Zoom and hover to explore"
        )
        fig = build_chart(d["hist"], la["ticker"])
        st.plotly_chart(fig, use_container_width=True)

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
            if st.button(f"⭐ Add {la['raw'].title()} to Wishlist",
                         type="primary", use_container_width=True):
                if add_to_wishlist(
                        la["raw"].title(), la["ticker"], d["price"]):
                    st.success("✅ Added to wishlist!")
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

        st.subheader("💬 Ask AI Anything")
        st.caption(
            f"Ask any follow-up question about {la['raw'].title()} "
            "or Indian markets in general."
        )

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if question := st.chat_input(
            f"Ask something about {la['raw'].title()}..."
        ):
            st.session_state.chat_history.append(
                {"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_ai(question, la)
                st.markdown(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer})

        st.divider()
        st.caption(
            f"Generated {datetime.now().strftime('%d %b %Y, %I:%M %p')} IST"
            "  |  Not SEBI-registered financial advice."
        )

    else:
        st.info(
            "Enter a company name above and click **Run Analysis ▶**\n\n"
            "Or click **Analyse ▶** on any stock in the "
            "🏠 Market Overview tab."
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — P&L SIMULATOR
# ════════════════════════════════════════════════════════════════════════════════
with tab_simulator:
    st.subheader("🧮 Historical P&L Simulator")
    st.caption(
        "See what a past investment would be worth today, "
        "compared against Nifty 50."
    )

    sim1, sim2 = st.columns(2)
    with sim1:
        sim_company = st.text_input(
            "Company Name or Ticker",
            placeholder="e.g. Wipro, RELIANCE.NS",
            key="sim_company",
        )
        sim_exchange = st.radio("Exchange", ["NSE", "BSE"],
                                horizontal=True, key="sim_exchange")
    with sim2:
        sim_amount = st.number_input(
            "Amount Invested (₹)", min_value=1000.0,
            step=1000.0, value=10000.0,
        )
        sim_date = st.date_input(
            "Investment Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now() - timedelta(days=30),
        )

    sim_btn = st.button("Calculate Returns 📊", type="primary",
                        use_container_width=True)

    if sim_btn:
        if not sim_company.strip():
            st.warning("Please enter a company name or ticker.")
        else:
            raw_sim = sim_company.strip()
            if raw_sim.upper().endswith((".NS", ".BO")):
                sim_ticker = raw_sim.upper()
            elif raw_sim.strip().isdigit():
                sim_ticker = f"{raw_sim.strip()}.BO"
            else:
                with st.spinner("Looking up ticker..."):
                    sim_ticker = resolve_ticker(raw_sim, sim_exchange)

            if not sim_ticker:
                st.error(f"Could not find ticker for **{raw_sim}**.")
            else:
                with st.spinner("Calculating returns..."):
                    result = simulate_pnl(
                        sim_ticker, sim_amount,
                        sim_date.strftime("%Y-%m-%d"),
                    )

                if not result:
                    st.error(
                        "Could not fetch historical data. "
                        "Try a different date or ticker."
                    )
                else:
                    st.subheader(
                        f"Results: ₹{sim_amount:,.0f} in "
                        f"{raw_sim.title()} on "
                        f"{sim_date.strftime('%d %b %Y')}"
                    )

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Invested",      f"₹{result['invested']:,.2f}")
                    r2.metric("Current Value", f"₹{result['curr_value']:,.2f}",
                              f"₹{result['pnl']:+,.2f}")
                    r3.metric("Total Return",  f"{result['pnl_pct']:+.2f}%",
                              f"CAGR {result['cagr']:+.2f}%")
                    r4.metric(
                        "Nifty 50 Return (same period)",
                        f"{result['nifty_return']:+.2f}%",
                        f"{'Beat' if result['pnl_pct'] > result['nifty_return'] else 'Lagged'}"
                        f" Nifty by "
                        f"{abs(result['pnl_pct'] - result['nifty_return']):.2f}%",
                        delta_color=(
                            "normal" if result['pnl_pct'] > result['nifty_return']
                            else "inverse"
                        ),
                    )

                    d1, d2, d3 = st.columns(3)
                    d1.metric("Buy Price",     f"₹{result['buy_price']}")
                    d2.metric("Current Price", f"₹{result['sell_price']}")
                    d3.metric("Shares Bought", f"{result['shares']}")

                    st.subheader("📈 Portfolio Growth vs Nifty 50")
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Scatter(
                        x=result["stock_norm"].index,
                        y=result["stock_norm"].values.flatten(),
                        name=raw_sim.title(),
                        line=dict(color="#26a69a", width=2),
                    ))
                    if result["nifty_norm"] is not None:
                        fig_sim.add_trace(go.Scatter(
                            x=result["nifty_norm"].index,
                            y=result["nifty_norm"].values.flatten(),
                            name="Nifty 50",
                            line=dict(color="#f39c12", width=2, dash="dot"),
                        ))
                    fig_sim.update_layout(
                        height=350,
                        yaxis_title=f"Value of ₹{sim_amount:,.0f} invested",
                        xaxis_title="Date",
                        legend=dict(orientation="h"),
                        margin=dict(l=0, r=0, t=20, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    fig_sim.update_xaxes(
                        gridcolor="rgba(128,128,128,0.15)")
                    fig_sim.update_yaxes(
                        gridcolor="rgba(128,128,128,0.15)")
                    st.plotly_chart(fig_sim, use_container_width=True)

                    st.caption(
                        f"Based on {result['days']} days of holding "
                        f"({sim_date.strftime('%d %b %Y')} to today).  "
                        "Past performance is not a guarantee of future returns."
                    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — MUTUAL FUNDS
# ════════════════════════════════════════════════════════════════════════════════
with tab_mf:
    st.subheader("💰 Mutual Fund Advisor")
    st.caption("Get 10 personalised fund recommendations based on your profile.")

    mf1, mf2 = st.columns(2)
    with mf1:
        inv_type = st.radio("Investment type", ["SIP", "Lump Sum"],
                            horizontal=True)
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
# TAB 5 — WISHLIST
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
            "Or click **Analyse ▶** on any stock in the "
            "🏠 Market Overview tab."
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
                    live_hist = clean_yf_df(
                        yf.Ticker(item["ticker"]).history(period="1d"))
                    if not live_hist.empty:
                        live      = round(
                            float(pd.Series(live_hist["Close"].to_numpy().flatten().astype(float)).iloc[-1]), 2)
                        delta     = round(live - item["price"], 2)
                        delta_pct = round(
                            (delta / item["price"]) * 100, 2)
                        st.metric(
                            "Live Price", f"₹{live}",
                            f"₹{delta:+.2f} ({delta_pct:+.2f}%) since added",
                        )
                except Exception:
                    st.caption("Live price unavailable")
            with col_d:
                if st.button("🗑️", key=f"del_{i}",
                             help="Remove from wishlist"):
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
        to_email  = st.text_input("Your email address",
                                   placeholder="you@gmail.com")
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
                    ok = send_wishlist_email(
                        to_email, st.session_state.wishlist)
                if ok:
                    st.success(f"✅ Wishlist sent to {to_email}!")
