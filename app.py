import streamlit as st
import yfinance as yf
import pandas as pd
from ml_utils import generate_features, load_global_model, predict_signals
from backtest import run_backtest, plot_backtest_chart, plot_cumulative_returns
from datetime import datetime, timedelta

st.set_page_config(page_title="Global AI Stock Signal Detector", layout="wide")
st.markdown("<h1 style='text-align: center;'>📈 AI Stock Signal Detector (Global Model)</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.header("Stock Configuration")
stock_symbol = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.today())

if st.sidebar.button("🔍 Run Analysis"):
    with st.spinner("Downloading stock data..."):
        try:
            data = yf.download(stock_symbol, start=start_date, end=end_date)
            if data.empty:
                st.error("No data found. Please check the ticker symbol or date range.")
                st.stop()
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    st.success(f"✅ Data fetched for {stock_symbol.upper()} from {start_date} to {end_date}")
    
    data = yf.download(stock_symbol, start=start_date, end=end_date)  # ✅ Download
    df = generate_features(data)                # ✅ 2. Add indicators/features
    model = load_global_model()                 # ✅ 3. Load trained model
    df = predict_signals(model, df)             # ✅ 4. Predict signals

    st.subheader("📌 Latest Predicted Signals")
    st.dataframe(df[['Close', 'signal']].tail(10), use_container_width=True)

    if 'Close' not in df.columns or 'signal' not in df.columns:
        st.error("❌ Required columns 'Close' and 'signal' not found in the data.")
        st.stop()
    backtest_result = run_backtest(df)

    with st.expander("📉 Backtest Performance Summary", expanded=True):
        st.markdown("### ✅ Strategy Metrics")
        st.write(f"**Total Trades:** {backtest_result['trades']}")
        st.write(f"**Average Return per Trade:** {backtest_result['avg_return']:.2f}%")
        st.write(f"**Total Return:** {backtest_result['total_return']:.2f}%")

    with st.expander("📈 Signal Overlay Chart"):
        fig_overlay = plot_backtest_chart(df)
        st.pyplot(fig_overlay)

    with st.expander("📊 Cumulative Return Chart"):
        fig_return = plot_cumulative_returns(backtest_result['returns'])
        st.pyplot(fig_return)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<center>🔹 Built with ❤️ by <b>Manzar</b> for Smart Traders</center>", unsafe_allow_html=True)