import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import quandl
from sklearn.linear_model import LinearRegression

@st.cache_data()
def load_stock_data(symbol, start_date, end_date):
    quandl.ApiConfig.api_key = 'W4_Y1GsfYpohRrRJg9ub'
    stock_data = quandl.get(f'NSE/{symbol}', start_date=start_date, end_date=end_date).copy()
    return stock_data

def calculatePredictions(model, data, X):
    st.subheader('Input Parameters')
    st.write('Use the sliders below to input future values to predict:')
    
    with st.container(border=4):
        # Create a two-column layout
        col1, col2 = st.columns(2)

        future_values = {}
        for feature in X.columns:
            future_values[feature] = col1.slider(f'{feature}', min_value=data[feature].min(), max_value=data[feature].max(), value=data[feature].median())
        
        future_values_df = pd.DataFrame([future_values])
        future_predictions = model.predict(future_values_df)

        col2.subheader('Predictions')
        col2.write(f'## {float(future_predictions[0]):.3f}')

        # Display heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        col2.pyplot(plt)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    mae = np.mean(np.abs(predictions - y_test))
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Squared Error', 'Mean Absolute Error', 'Root Mean Squared Error', 'R2 Score'],
        'Value': [mean_squared_error(y_test, predictions), mae, rmse, r2_score(y_test, predictions)]
    })
    st.table(metrics_df)

def getsidebardata():
    nse_stocks = [
    'TCS', 'INFY', 'RELIANCE', 'WIPRO', 'HDFC', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'AXISBANK',
    'BAJAJ-AUTO', 'BAJAJFINSV', 'BAJFINANCE', 'BHARTIARTL', 'BPCL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 
    'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HEROMOTOCO', 'HINDALCO', 
    'HINDPETRO', 'HDFCLIFE', 'HINDZINC', 'HDFC']
    
    symbol = st.sidebar.selectbox('Select Stock Symbol', nse_stocks, index=0)
    with st.container(border=2):
            start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2015-12-01'))
            end_date = st.sidebar.date_input('End Date', pd.to_datetime('2018-12-01'))

    return symbol, start_date, end_date

def plotstonks(data):
    st.markdown("# Data analysis")

    with st.expander("Stock Price Trends", expanded=True):
        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label='Close Price', color='blue')
        plt.plot(data['Open'], label='Open Price', color='green')
        plt.plot(data['High'], label='High Price', color='red')
        plt.plot(data['Low'], label='Low Price', color='orange')
        plt.title('Stock Price Trends')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)
    
    # Line plot of 'Close' prices over time
    with st.expander('Close Price Over Time', expanded=True):
        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label='Close Price', color='blue')
        plt.title('Close Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)
    
    # Histogram of 'Close' prices
    with st.expander('Histogram of Close Prices', expanded=True):
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Close'], kde=True, color='green')
        plt.title('Histogram of Close Prices')
        plt.xlabel('Close Price')
        plt.ylabel('Frequency')
        st.pyplot(plt)
    
    # Box plot of 'Close' prices by month
    with st.expander('Box Plot of Close Prices by Month', expanded=True):
        data['Month'] = data.index.month
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Month', y='Close', data=data, palette='Set3')
        plt.title('Box Plot of Close Prices by Month')
        plt.xlabel('Month')
        plt.ylabel('Close Price')
        st.pyplot(plt)