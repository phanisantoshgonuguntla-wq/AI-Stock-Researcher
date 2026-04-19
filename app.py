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

# --- 4. THE SMART "NAME-TO-TICKER" ENGINE (STUCK-PROOF VERSION) ---
if analyze_btn:
    if not user_input:
        st.warning("Please enter a company name.")
    else:
        with st.status("🔍 Starting AI Research...", expanded=True) as status:
            try:
                # Part A: The Ticker Finder (Fast AI Step)
                status.write("Finding NSE/BSE Symbol...")
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                search_tool = TavilySearchTool()

                finder = Agent(
                    role='Stock Symbol Specialist',
                    goal=f'Identify the NSE ticker for {user_input}',
                    backstory='You are a fast database. You only return the symbol.',
                    llm=llm
                )
                
                find_task = Task(
                    description=f"What is the official NSE symbol for {user_input}? Example: 'Reliance' -> 'RELIANCE.NS'.",
                    expected_output='Just the symbol (e.g. TICKER.NS)',
                    agent=finder
                )
                
                ticker_result = Crew(agents=[finder], tasks=[find_task]).kickoff()
                ticker = str(ticker_result).strip().upper()
                status.write(f"✅ Symbol Found: {ticker}")

                # Part B: ULTRA-FAST PRICE FETCH (Avoids .info hang)
                status.write("Fetching live price...")
                asset = yf.Ticker(ticker)
                
                # Fetching 5 days of history is much faster and more reliable than .info
                hist = asset.history(period="5d")
                if hist.empty:
                    # Fallback for BSE or non-standard tickers
                    st.error(f"Data not found for {ticker}. Try adding '.NS' to your search.")
                    st.stop()
                
                curr_price = round(hist['Close'].iloc[-1], 2)
                prev_price = round(hist['Close'].iloc[-2], 2)
                change = round(curr_price - prev_price, 2)

                # Part C: MONEYCONTROL SEARCH
                status.write("Checking Moneycontrol news...")
                # include_domains keeps the search focused and fast
                money_search = TavilySearchTool(
                    include_domains=["moneycontrol.com", "economictimes.indiatimes.com"]
                )

                advisor = Agent(
                    role='Financial Analyst',
                    goal=f'Quick report on {user_input}',
                    backstory='You focus on recent news and technicals.',
                    tools=[money_search],
                    llm=llm,
                    allow_delegation=False  # Crucial for speed
                )

                task = Task(
                    description=f"Analyze {user_input} ({ticker}) at ₹{curr_price}. Check recent news and give a 1-paragraph advice.",
                    expected_output='A clear Buy/Sell/Hold rating with reasons.',
                    agent=advisor
                )

                result = Crew(agents=[advisor], tasks=[task]).kickoff()
                status.update(label="Analysis Complete!", state="complete", expanded=False)

                # --- 5. THE OUTPUT ---
                st.subheader(f"Equity Analysis: {user_input}")
                st.metric("Price", f"₹{curr_price}", delta=f"₹{change}")
                st.markdown(result.raw)

            except Exception as e:
                st.error(f"Timeout or Error: {e}")
                st.info("Tip: If it hangs, try refreshing and typing the full name (e.g. 'HDFC Bank Ltd').")
