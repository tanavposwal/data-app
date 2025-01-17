import streamlit as st
import yfinance as yf

st.header("Simple Stock Explorer", divider="rainbow")

symbl = "GOOGL"
st.text_input("Symbol", symbl)

if st.button("Get Data", type="secondary"):
    tickerdata = yf.Ticker(symbl)
    tickerdf = tickerdata.history(period="1d", start="2010-5-31")

    st.write("Closing")
    st.line_chart(tickerdf.Close)

st.write("A small data app for stocks enthusiasts")
