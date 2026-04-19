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

# --- 4. CORE ENGINE ---
if analyze_btn:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.status("🔍 Scanning Market Data...", expanded=True) as status:
            try:
                # A. Reliable Price Fetching
                asset = yf.Ticker(ticker)
                fast = asset.fast_info
                
                curr_price = fast.get('last_price') or asset.info.get('currentPrice') or \
                             asset.info.get('navPrice') or asset.info.get('previousClose', 0)
                
                high_52 = fast.get('year_high') or asset.info.get('fiftyTwoWeekHigh', 0)
                low_52 = fast.get('year_low') or asset.info.get('fiftyTwoWeekLow', 0)
                asset_name = asset.info.get('longName', ticker)

                if curr_price == 0:
                    st.error(f"Could not find {ticker}. Please check the symbol.")
                    st.stop()

                # B. AI Agent Configuration
                status.write("🤖 AI Analyst is researching latest news...")
                search_tool = TavilySearchTool()
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

                advisor = Agent(
                    role='Senior Indian Market Analyst',
                    goal=f'Provide clear Buy/Sell/Hold advice for {asset_name}',
                    backstory='Expert in Indian stock trends, regulatory news, and 52-week technicals.',
                    tools=[search_tool],
                    llm=llm,
                    verbose=True
                )

                portfolio_msg = f"User bought at ₹{buy_price} and is holding." if is_holding else "User is looking to enter a new position."
                
                task = Task(
                    description=f'''
                    Analyze {asset_name} ({ticker}). 
                    Market Stats: Price ₹{curr_price}, 52W High ₹{high_52}, 52W Low ₹{low_52}.
                    User Context: {portfolio_msg}
                    Research the latest 3 news items from Indian financial media and provide a clear recommendation.''',
                    expected_output='A professional report with price metrics, news impact, and final advice.',
                    agent=advisor
                )

                crew = Crew(agents=[advisor], tasks=[task])
                result = crew.kickoff()
                status.update(label="Analysis Complete!", state="complete", expanded=False)

                # --- 5. RESULTS DISPLAY ---
                st.subheader(f"Strategy Report: {asset_name}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Current", f"₹{curr_price}")
                col2.metric("52W High", f"₹{high_52}")
                col3.metric("52W Low", f"₹{low_52}")
                
                st.markdown("---")
                st.markdown(result.raw)

            except Exception as e:
                st.error(f"Unexpected Error: {e}")
