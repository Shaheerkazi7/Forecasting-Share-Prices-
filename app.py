import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

# Load the model
model = load_model('reliance_lstm_model.h5')

# Load the data
data = pd.read_csv('reliance_stock_data.csv', parse_dates=['Date'], index_col='Date')

# Use only the 'Close' prices for prediction
close_data = data['Close'].values.reshape(-1, 1)

# Scale the data to the range (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

# Function to create dataset for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Function to forecast next days
def forecast_next_days(model, data, time_step, n_days):
    temp_input = list(data[-time_step:].flatten())
    output = []
    for _ in range(n_days):
        if len(temp_input) > time_step:
            temp_input = temp_input[1:]
        input_data = np.array(temp_input).reshape(1, time_step, 1)
        prediction = model.predict(input_data)
        output.append(prediction[0, 0])
        temp_input.append(prediction[0, 0])
    return np.array(output)

# Streamlit app
st.title('Reliance Industries Stock Price Prediction')

# Date input
start_date = st.date_input('Start Date', value=datetime.date(2024, 4, 24))
end_date = st.date_input('End Date', value=datetime.date(2025, 4, 24))

# Predict button
if st.button('Predict'):
    # Number of days to predict
    n_days = (end_date - start_date).days + 1  # Include both start and end dates
    
    # Convert the start date to a pd.Timestamp for comparison
    last_historical_date = data.index[-1].date()
    if start_date <= last_historical_date:
        forecast_start_date = last_historical_date + datetime.timedelta(days=1)
    else:
        forecast_start_date = start_date
    
    # Calculate the days to forecast from the forecast_start_date to the end_date
    forecast_days = (end_date - forecast_start_date).days + 1
    
    # Forecast the prices
    forecasted_prices = forecast_next_days(model, scaled_data, time_step=100, n_days=forecast_days)
    forecasted_prices = scaler.inverse_transform(forecasted_prices.reshape(-1, 1))
    
    # Generate the dates for the forecasted period
    forecast_dates = pd.date_range(start=forecast_start_date, periods=forecast_days, freq='D')
    
    # Create a new DataFrame for the forecasted prices
    forecast_df = pd.DataFrame(forecasted_prices, index=forecast_dates, columns=['Forecasted Close'])
    
    # Display forecasted prices in a box
    st.subheader('Forecasted Prices for the Selected Period')
    st.write(forecast_df)
    
    # Plot the forecasted prices along with the historical data using Plotly
    fig = go.Figure()
    
    # Add historical data up to the forecast_start_date
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Prices'))
    
    # Add forecasted data
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecasted Close'], mode='lines', name='Forecasted Prices'))
    
    # Update layout
    fig.update_layout(
        title='Reliance Industries Stock Price Forecast',
        xaxis_title='Date',
        yaxis_title='Close Price (INR)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig)


