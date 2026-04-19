import streamlit as st
import os
import yfinance as yf
from crewai import Agent, Task, Crew
from crewai_tools import TavilySearchTool
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. SECRETS CONFIGURATION ---
st.set_page_config(page_title="2026 AI Wealth Advisor", layout="wide")

# This pulls keys from the Streamlit Dashboard "Secrets" section
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except Exception:
    st.error("❌ Missing Secrets! Go to Streamlit Settings -> Secrets and add GOOGLE_API_KEY and TAVILY_API_KEY.")
    st.stop()

# --- 2. APP UI ---
st.title("🇮🇳 Indian Stock & Portfolio Advisor")
st.markdown("Automated analysis for **NSE**, **BSE**, and **Mutual Funds**.")

with st.sidebar:
    st.header("Portfolio Input")
    ticker = st.text_input("Ticker Symbol", placeholder="e.g. HDFCBANK.NS or 500180.BO")
    st.caption("Tip: Use .NS for NSE, .BO for BSE.")
    
    is_holding = st.checkbox("I am holding this stock")
    buy_price = 0.0
    if is_holding:
        buy_price = st.number_input("Purchase Price (₹)", min_value=0.0, step=1.0)
    
    analyze_btn = st.button("Analyze Stock", type="primary")

# --- 3. CORE LOGIC ---
if analyze_btn:
    if not ticker:
        st.warning("Please enter a ticker symbol first.")
    else:
        with st.status("Fetching live data...", expanded=True) as status:
            try:
                # STEP A: Robust Price Fetching (Fixes HDFCBANK.NS error)
                asset = yf.Ticker(ticker)
                
                # Using fast_info for better reliability in 2026 cloud environments
                fast = asset.fast_info
                
                # Try multiple fields in case one is empty
                curr_price = fast.get('last_price') or \
                             asset.info.get('currentPrice') or \
                             asset.info.get('navPrice') or \
                             asset.info.get('previousClose') or 0
                
                high_52 = fast.get('year_high') or asset.info.get('fiftyTwoWeekHigh', 0)
                low_52 = fast.get('year_low') or asset.info.get('fiftyTwoWeekLow', 0)
                asset_name = asset.info.get('longName', ticker)

                if curr_price == 0:
                    st.error(f"Could not fetch price for {ticker}. Try adding .NS or .BO.")
                    st.stop()

                # STEP B: AI Analysis
                status.write("Running AI Agent analysis...")
                search_tool = TavilySearchTool()
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

                advisor = Agent(
                    role='Senior Indian Market Analyst',
                    goal=f'Give a clear Buy/Sell/Hold advice for {asset_name}',
                    backstory='Expert in Indian markets, focused on NSE/BSE trends.',
                    tools=[search_tool],
                    llm=llm,
                    verbose=True
                )

                portfolio_msg = f"User holds at ₹{buy_price}." if is_holding else "User is looking to buy."
                
                task = Task(
                    description=f'''Analyze {asset_name} ({ticker}). 
                    Current Price: ₹{curr_price}, 52W High: ₹{high_52}, 52W Low: ₹{low_52}.
                    {portfolio_msg} 
                    Research latest news and give a report with a clear recommendation.''',
                    expected_output='A detailed report with a recommendation.',
                    agent=advisor
                )

                crew = Crew(agents=[advisor], tasks=[task])
                result = crew.kickoff()
                status.update(label="Analysis Done!", state="complete", expanded=False)

                # --- 4. DISPLAY RESULTS ---
                st.subheader(f"Results for {asset_name}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"₹{curr_price}")
                c2.metric("52W High", f"₹{high_52}")
                c3.metric("52W Low", f"₹{low_52}")
                
                st.markdown("---")
                st.markdown(result.raw)

            except Exception as e:
                st.error(f"Error: {e}")
