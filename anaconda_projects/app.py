import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as data
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries  # Alpha Vantage API
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Streamlit UI setup
st.title('Stock Trend Prediction')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
start = '2010-01-01'
end = '2019-12-31'

# Alpha Vantage API Key (replace 'your_alpha_vantage_api_key' with your actual API key)
api_key = 'your_alpha_vantage_api_key'

# Create a TimeSeries object from Alpha Vantage
ts = TimeSeries(key=api_key, output_format='pandas')

# Fetch the stock data using Alpha Vantage (daily data for the given ticker)
try:
    data, meta_data = ts.get_daily(symbol=user_input, outputsize='full')
    
    # Check if the data was returned properly
    if data.empty:
        st.error("No data fetched for the given symbol.")
    else:
        st.subheader(f"Data for {user_input} from 2010-2019")
        st.write(data.describe())

        # Visualizing the Closing Price vs Time chart
        st.subheader('Closing Price vs Time chart')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(data['4. close'])  # Closing price column from Alpha Vantage response
        plt.title(f"{user_input} Closing Price")
        st.pyplot(fig)

        # Visualization with 100-day Moving Average (MA)
        st.subheader('Closing Price vs Time chart with 100MA')
        ma100 = data['4. close'].rolling(100).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ma100)
        plt.plot(data['4. close'])
        st.pyplot(fig)

        # Visualization with 100-day and 200-day Moving Averages (MA)
        st.subheader('Closing Price vs Time chart with 100MA & 200MA')
        ma200 = data['4. close'].rolling(200).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ma100, 'r', label='100MA')
        plt.plot(ma200, 'g', label='200MA')
        plt.plot(data['4. close'], 'b', label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        # Splitting the data into training and testing sets
        data_training = pd.DataFrame(data['4. close'][0:int(len(data) * 0.70)])
        data_testing = pd.DataFrame(data['4. close'][int(len(data) * 0.70):])

        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Load pre-trained model
        try:
            model = load_model('keras_model.keras')
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        # Prepare input data for prediction
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions using the model
        y_predicted = model.predict(x_test)

        # Rescale the predictions and actual values
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Plotting the results
        st.subheader('Predictions vs Original')
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

except Exception as e:
    st.error(f"An error occurred: {e}")
