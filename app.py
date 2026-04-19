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

# --- 3. SMART UI SETUP (REPLACE YOUR OLD STEP 3) ---
with st.sidebar:
    st.header("Search by Name")
    # This now accepts "Tata Motors" or "Reliance" instead of tickers
    user_input = st.text_input("Enter Company Name", placeholder="e.g. HDFC Bank, Infosys")
    
    is_holding = st.checkbox("I am holding this stock")
    buy_price = 0.0
    if is_holding:
        buy_price = st.number_input("Purchase Price (₹)", min_value=0.0)
    
    analyze_btn = st.button("Run AI Analysis", type="primary")

# --- 4. SMART SEARCH ENGINE (REPLACE YOUR OLD STEP 4) ---
if analyze_btn:
    if not user_input:
        st.warning("Please enter a company name.")
    else:
        with st.status("🔍 Converting Name to Ticker...", expanded=True) as status:
            try:
                # Part A: The Translator (Agent finds the Ticker)
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                search_tool = TavilySearchTool()

                finder = Agent(
                    role='Stock Symbol Finder',
                    goal=f'Find the NSE ticker for {user_input}',
                    backstory='You are a database of NSE/BSE symbols. You only return the symbol.',
                    tools=[search_tool],
                    llm=llm
                )
                
                find_task = Task(
                    description=f"Find the official NSE symbol for {user_input}. Example: 'Reliance' -> 'RELIANCE.NS'.",
                    expected_output='Just the symbol string.',
                    agent=finder
                )
                
                ticker = str(Crew(agents=[finder], tasks=[find_task]).kickoff()).strip()
                status.write(f"Using Ticker: **{ticker}**")

                # Part B: Price & Analysis
                asset = yf.Ticker(ticker)
                hist = asset.history(period="1mo")
                curr_price = round(hist['Close'].iloc[-1], 2)

                status.write("Consulting Moneycontrol & Economic Times...")
                advisor = Agent(
                    role='Moneycontrol Analyst',
                    goal=f'Analyze {user_input} ({ticker})',
                    backstory='Expert in Indian markets.',
                    tools=[search_tool],
                    llm=llm
                )

                task = Task(
                    description=f"Analyze {user_input} at ₹{curr_price}. Check Moneycontrol for news.",
                    expected_output='A professional Buy/Sell/Hold report.',
                    agent=advisor
                )

                result = Crew(agents=[advisor], tasks=[task]).kickoff()
                status.update(label="Analysis Done!", state="complete", expanded=False)

                # --- 5. THE OUTPUT ---
                st.subheader(f"Strategy Report: {user_input}")
                st.metric("Live Price", f"₹{curr_price} ({ticker})")
                st.markdown(result.raw)

            except Exception as e:
                st.error("Make sure to use a clear company name (e.g. 'Tata Motors' instead of just 'Tata').")
