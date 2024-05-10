import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from modules.utils import load_stock_data, train_model, evaluate_model, plotstonks, getsidebardata, calculatePredictions

def model_page():
    st.title('Market analysis')
    st.write("Select a NSE stock from the sidebar and predict closing price of the stock using a linear regression model. Source of data: **_quandl_** public API")
    st.sidebar.image("assets/stonks.jpeg", use_column_width=True)
    st.sidebar.divider()
    st.divider()

    symbol, start_date, end_date = getsidebardata()
    data = load_stock_data(symbol, start_date, end_date)

    start_price = data.iloc[0]['Close']
    end_price = data.iloc[-1]['Close']
    profit = end_price - start_price
    profit_percentage = (profit / start_price) * 100
    
    st.subheader('Data')
    st.write(data.head(100))
    
    X = data[['Open', 'High', 'Low', 'Last', 'Total Trade Quantity', 'Turnover (Lacs)']]
    y = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)

    st.subheader('Overall Stock Performance')
    st.write(f"Start Date: {start_date}, End Date: {end_date}")
    
    profit_data = pd.DataFrame({
        'Metric': ['Profit', 'Profit Percentage'],
        'Value': [profit, f"{profit_percentage:.2f}%"]
    })
    st.table(profit_data)

    st.divider()
    
    st.subheader('Model Evaluation')
    evaluate_model(model, X_test, y_test)
    calculatePredictions(model, data, X)
    plotstonks(data)





def home_page():
    st.title("📈 stonks")
    st.subheader(
        "just another _stonks_ prediction streamlit application")

    st.sidebar.image("assets/notstonks.jpeg", use_column_width=True)
    st.divider()

    with st.container():
        st.header('How to use the site:')
        st.markdown(
            """
            1. Select the stonk you want to analyse.
            1. We get the stock data using **quandl** public api to get the details of the selected stonk.
            1. Preview the visualisation of the stonk history.
            1. Predict the stonk closing prise based on the linear regression model trained using scikit learn.
            """
        )

    st.markdown("""

    ## **How It's Built**

    Stockastic is built with these core frameworks and modules:

    - **Streamlit** - To create the web app UI and interactivity 
    - **Scikit Learn** - To build the ARIMA time series forecasting model
    - **Quandil** - To fetch financial data
    - **seaborn and matplotlib** - To create interactive financial charts

    The app workflow is:

    1. User selects a stock ticker
    2. Historical data is fetched with Quandil
    3. Linear regression model is trained on the data 
    4. Model makes price forecasts absed on the parameters user provide
    5. Results are plotted with matplotlib and seaborn

    """
    )

    st.divider()

    st.title("FAQs")

    with st.expander("## How do I get the stock data?"):
        st.write('''
        Using a public API provided by [Quandl](https://www.quandl.com/), the application fetches historical stock data based on the selected stock symbol and date range.
                 
        ```python
        @st.cache_data()
        def load_stock_data(symbol, start_date, end_date):
            quandl.ApiConfig.api_key = QUANDIL_API_KEY
            stock_data = quandl.get(f'NSE/{symbol}', start_date=start_date, end_date=end_date).copy()
            return stock_data
        ```
        ''')
    with st.expander("## What machine learning model is used for prediction?"):
        st.write('''
            The application employs a Linear Regression model from scikit-learn library for stock price prediction. This model is trained on the provided historical stock data to make future price predictions.
        ''')
    with st.expander("## How accurate are the predictions?"):
        st.write('''
            The accuracy of the predictions is evaluated using Mean Squared Error (MSE) and R-squared (R2) score metrics. MSE measures the average squared difference between the actual and predicted values, while R2 score indicates the proportion of the variance in the dependent variable (stock price) that is predictable from the independent variables (features). The evaluation results are displayed in the application.
        ''')
    with st.expander("## Can I customize the input values for prediction?"):
        st.write('''
            Yes, you can customize the input values for prediction using sliders provided in the "Predictions" section of the application. Simply adjust the sliders for features such as Open, High, Low, Last, Total Trade Quantity, and Turnover (Lacs) to input the desired future values for prediction.
        ''')
    with st.expander("## What does the histogram of close prices indicate?"):
        st.write('''
            The "Histogram of Close Prices" provides a visual representation of the distribution of Close Prices for the selected stock. The histogram displays the frequency of different Close Price ranges, allowing you to see the distribution pattern and understand the typical range of Close Prices.
        ''')

    st.markdown("""
    My kaggle [notebook](https://www.kaggle.com/code/decentralized/linear-regression-in-stock-market) for the project.
                
    ---
                
    10 May 2024
    
    Stonks project was created by [aditya](https://adimail.github.io/) under the term work for AI course, btech semester 4. 
    """)