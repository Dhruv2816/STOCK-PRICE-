import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Load the trained model
model = load_model(r'MODEL/Stock Predictions Model.keras')

# Streamlit header
st.header('Stock Price Predictor')

# Input for stock symbol
stock = st.text_input('ENTER THE SYMBOL FOR STOCK PRICE', 'TCS.NS')
import datetime

# Dynamically fetch today's date
end = datetime.datetime.today().strftime('%Y-%m-%d')
# end = "2024-01-19"
# Dynamically fetch the start date (first available date of the stock)
stock_info = yf.Ticker(stock)
start = stock_info.history(period="max").index[0].strftime('%Y-%m-%d')
# Fetch stock data
data = yf.download(stock, start, end)

# Split data into train and test
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.70)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.70):])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Prepare test data
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Moving averages
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

# Prepare data for prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Predict and scale back to original values
predict = model.predict(x)
scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)

# Predict the next 5 days
future_predictions = []
last_100_days = data_test_scale[-100:].reshape(1, 100, 1)

for _ in range(5):
    pred = model.predict(last_100_days)  # Predict the next value
    future_predictions.append(pred[0, 0])  # Store the prediction
    # Append the prediction and slide the window
    pred_reshaped = pred.reshape(1, 1, 1)  # Reshape prediction
    last_100_days = np.append(last_100_days[:, 1:, :], pred_reshaped, axis=1)

# Scale back the predictions
future_predictions = np.array(future_predictions) * scale[0]

# Display future predictions in a table
st.subheader('Predicted Prices for the Next 5 Days')
future_df = pd.DataFrame({
    'Day': [f'Day {i+1}' for i in range(5)],
    'Predicted Price': future_predictions
})
st.table(future_df)
