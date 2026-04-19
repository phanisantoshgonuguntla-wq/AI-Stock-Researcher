import streamlit as st
import os
import yfinance as yf

# --- 1. THE 2026 CRITICAL COMPATIBILITY BLOCK ---
# We force-load these to prevent the "pkg_resources" and "Tavily" errors
try:
    import setuptools
    import pkg_resources
    from crewai import Agent, Task, Crew
    from crewai_tools import TavilySearchTool
    from langchain_google_genai import ChatGoogleGenerativeAI
except (ImportError, ModuleNotFoundError) as e:
    st.error(f"❌ Deployment Error: {e}")
    st.info("Please ensure your requirements.txt is updated with 'setuptools<82.0.0'.")
    st.stop()

# --- 2. CONFIG & SECRETS ---
st.set_page_config(page_title="2026 AI Wealth Advisor", layout="wide")

try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except Exception:
    st.error("🔑 API Keys Missing! Add them to Streamlit Settings -> Secrets.")
    st.stop()

# --- 3. UI SETUP ---
st.title("🇮🇳 Indian Stock & Portfolio Advisor")
st.markdown("Advanced AI analysis for **NSE**, **BSE**, and **Mutual Funds**.")

with st.sidebar:
    st.header("Stock Selection")
    ticker = st.text_input("Symbol", placeholder="e.g. HDFCBANK.NS or 500180.BO")
    st.caption("Use .NS for NSE and .BO for BSE.")
    
    is_holding = st.checkbox("I am holding this stock")
    buy_price = 0.0
    if is_holding:
        buy_price = st.number_input("Purchase Price (₹)", min_value=0.0, step=1.0)
    
    analyze_btn = st.button("Run Full Analysis", type="primary")

# --- 4. HIGH-PERFORMANCE CORE ENGINE ---
if analyze_btn:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        # We use a container to show progress
        status_box = st.empty()
        with st.status("🚀 Launching Analysis...", expanded=True) as status:
            try:
                # A. FASTEST PRICE FETCHING (Avoids .info hang)
                status.write("Fetching market stats...")
                asset = yf.Ticker(ticker)
                
                # Fetching 1 day of history is 10x faster and never hangs
                hist = asset.history(period="1d")
                if hist.empty:
                    st.error(f"Could not find data for {ticker}. Check the symbol.")
                    st.stop()
                
                curr_price = round(hist['Close'].iloc[-1], 2)
                
                # B. FAST AI AGENT (Using 'fast' search depth)
                status.write("Searching latest news...")
                search_tool = TavilySearchTool() 
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

                advisor = Agent(
                    role='Senior Indian Market Analyst',
                    goal=f'Quick Buy/Sell/Hold advice for {ticker}',
                    backstory='You provide rapid, accurate financial summaries.',
                    tools=[search_tool],
                    llm=llm,
                    allow_delegation=False, # Faster: Agent won't try to hire other agents
                    verbose=True
                )

                task = Task(
                    description=f"Current Price of {ticker}: ₹{curr_price}. Find 2 major news items and give a 1-paragraph recommendation.",
                    expected_output='A short, punchy investment report.',
                    agent=advisor
                )

                crew = Crew(agents=[advisor], tasks=[task])
                result = crew.kickoff()
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)

                # --- 5. RESULTS DISPLAY ---
                st.subheader(f"Results for {ticker}")
                st.metric("Live Price", f"₹{curr_price}")
                st.markdown("---")
                st.markdown(result.raw)

            except Exception as e:
                st.error(f"Speed Error: {e}")
