import streamlit as st
import os
import yfinance as yf
from crewai import Agent, Task, Crew
from crewai_tools import TavilySearchTool
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. CONFIG & SECRETS ---
st.set_page_config(page_title="2026 AI Wealth Advisor", layout="wide")

# Automatically pull keys from Streamlit Cloud Secrets
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except Exception:
    st.error("Missing Secrets! Please add GOOGLE_API_KEY and TAVILY_API_KEY in Streamlit Settings.")
    st.stop()

# --- 2. UI HEADER ---
st.title("🇮🇳 Indian Stock & Portfolio Advisor")
st.markdown("Automated analysis for **NSE**, **BSE**, and **Mutual Funds** using Gemini 2.5 Flash.")

# --- 3. SIDEBAR & INPUTS ---
with st.sidebar:
    st.header("Portfolio Status")
    ticker = st.text_input("Ticker Symbol", placeholder="e.g. RELIANCE.NS or 500325.BO")
    st.caption("Use .NS for NSE, .BO for BSE, or Fund ID for Mutual Funds.")
    
    is_holding = st.checkbox("I am holding this asset")
    buy_price = 0.0
    if is_holding:
        buy_price = st.number_input("Purchase Price (₹)", min_value=0.0, step=1.0)
    
    analyze_btn = st.button("Run Full AI Analysis", type="primary")

# --- 4. DATA FETCHING & AI LOGIC ---
if analyze_btn:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.status("Fetching market data and searching news...", expanded=True) as status:
            # A. Get Price Data
            try:
                asset = yf.Ticker(ticker)
                info = asset.info
                # Logic to handle both Stocks and Mutual Funds (NAV)
                curr_price = info.get('currentPrice') or info.get('navPrice') or info.get('previousClose', 0)
                high_52 = info.get('fiftyTwoWeekHigh', 0)
                low_52 = info.get('fiftyTwoWeekLow', 0)
                asset_name = info.get('longName', ticker)
            except Exception as e:
                st.error(f"Error fetching data for {ticker}. Ensure the symbol is correct.")
                st.stop()

            # B. Setup AI Agent
            # Note: search_tool initialized without 'topic' to bypass confirmation bugs
            search_tool = TavilySearchTool()
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

            advisor = Agent(
                role='Senior Indian Investment Strategist',
                goal=f'Analyze {asset_name} and provide a definitive Buy/Sell/Hold recommendation.',
                backstory='''You are an elite analyst for Indian markets. You combine technical levels 
                (52-week ranges) with fundamental news sentiment to give high-accuracy advice.''',
                tools=[search_tool],
                llm=llm,
                verbose=True
            )

            # C. Define Recommendation Logic
            portfolio_context = f"The user is holding this asset at a buy price of ₹{buy_price}." if is_holding else "The user does not currently own this asset."
            
            task = Task(
                description=f'''
                1. Asset: {asset_name} ({ticker}).
                2. Technicals: Current Price ₹{curr_price}, 52W High ₹{high_52}, 52W Low ₹{low_52}.
                3. Search for the latest news impact (SEBI, Global trends, Earnings).
                4. Context: {portfolio_context}
                5. Recommendation: 
                   - If holding: Provide a "HOLD" or "SELL/PROFIT BOOK" target.
                   - If not holding: Provide a "BUY", "WAIT", or "AVOID" recommendation.''',
                expected_output='A clear, formatted report including data summary, news insights, and a final recommendation.',
                agent=advisor
            )

            # D. Execution
            crew = Crew(agents=[advisor], tasks=[task])
            result = crew.kickoff()
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # --- 5. DISPLAY RESULTS ---
        st.subheader(f"Analysis Report: {asset_name}")
        
        # Performance Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"₹{curr_price}")
        col2.metric("52W High", f"₹{high_52}", delta=f"{round(((curr_price-high_52)/high_52)*100, 2)}% from peak")
        col3.metric("52W Low", f"₹{low_52}", delta=f"{round(((curr_price-low_52)/low_52)*100, 2)}% from bottom", delta_color="normal")

        st.markdown("---")
        st.markdown(result.raw)
