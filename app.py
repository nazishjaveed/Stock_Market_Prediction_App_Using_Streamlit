import streamlit as st
from prophet import Prophet
from datetime import datetime
import pandas as pd
import yfinance as yf

# Title of the app
st.title("Stock Market Prediction App")

# User input for the stock ticker
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL for Apple)", "AAPL")

# Date range selection
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Load data function with updated caching
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error(f"Failed to retrieve data for {ticker}. Please check the ticker symbol.")
        return None
    data.reset_index(inplace=True)
    return data

# Load and display data
data_load_state = st.text("Loading data...")
data = load_data(ticker, start_date, end_date)
data_load_state.text("Loading data... done!")

if data is not None and not data.empty:
    st.subheader("Raw Data")
    st.write(data.tail())

    # Plot raw data
    st.subheader("Closing Price over Time")
    st.line_chart(data['Close'])

    # Prepare the data for Prophet
    if 'Date' in data.columns and 'Close' in data.columns:
        df = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

        # Ensure the 'y' column is numeric and drop any rows with missing values
        try:
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df.dropna(subset=['y'], inplace=True)
        except TypeError as e:
            st.error(f"Data formatting error: {e}")
        
        # Train Prophet model
        st.subheader("Training the Model...")
        try:
            m = Prophet()
            m.fit(df)
            
            # Make future dataframe
            future = m.make_future_dataframe(periods=365)
            
            # Forecast the data
            forecast = m.predict(future)
            
            # Show and plot forecast
            st.subheader("Forecast Data")
            st.write(forecast.tail())
            
            st.subheader("Forecasted Data with Prophet")
            fig1 = m.plot(forecast)
            st.pyplot(fig1)
            
            st.subheader("Forecast Components")
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"An error occurred during model training: {e}")
else:
    st.warning("No data available. Please check the stock ticker and date range.")
