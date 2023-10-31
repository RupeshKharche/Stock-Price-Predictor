import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import os
from pymongo import *

# Connecting to database
mongoURL = 'mongodb://localhost:27017/'
dbName = 'stock-db'
mongoClient = MongoClient(mongoURL)

db = mongoClient.get_database(dbName)


st.set_page_config(page_title='Stock Price Prediction')
st.title("Stock Price Prediction")

# Ticker input from user
ticker = st.selectbox("Choose a ticker", ['AAPL', 'GOOG', 'BAJFINANCE.NS', 'INFY.NS'], placeholder='Select a ticker')
tickerLowerCase = str(ticker).lower()

collection = db.get_collection(tickerLowerCase)

# Function to fetch stock data
@st.cache_data
def fetchStockData(ticker):
	data = yf.Ticker(ticker).history(period='max')
	data = data.reset_index()
	data['Date'] = pd.to_datetime(data['Date'], format='dd-mm-yyyy')
	if(collection.count_documents({}) > 0 and collection.count_documents({}) < len(data)):
		collection.delete_many({})
		collection.insert_many(data.to_dict('records'))
		return data
	elif(collection.count_documents({}) == 0):
		collection.insert_many(data.to_dict('records'))
		return data
	else:
		return pd.DataFrame(collection.find())


data = fetchStockData(ticker)
# Displaying graph of the stock data
st.subheader(f'Close Prices Vs Date for {ticker}')
st.line_chart(data, x='Date', y='Close')

# Ticker dict
tickerDict = dict({'GOOG' : 'goog', 'AAPL' : 'aapl', 'INFY.NS' : 'infy', 'BAJFINANCE.NS' : 'baj'})

# Loading the saved model from the ticker name
modelName = tickerDict[ticker]

if(f'{modelName}.h5' in os.listdir(os.getcwd())):
	model = tf.keras.models.load_model(f'{modelName}.h5', compile=False)
	model.compile()

	scaler = MinMaxScaler()

	split = int(len(data) * 0.8)
	closePrices = data['Close'].iloc[split:].to_numpy().reshape(-1, 1)
	scaledClose = scaler.fit_transform(closePrices)

	# Use 7 days of close prices to predict the 8th day's close price
	WINDOW = 7

	# Creating windows and labels
	X = []
	y = []

	for i in range(WINDOW, len(scaledClose)):
		X.append(scaledClose[i - WINDOW:i])
		y.append(scaledClose[i])

	X, y = np.array(X), np.array(y)

	X = np.reshape(X, (X.shape[0], X.shape[1], 1))

	# Predicting the stock price
	predictedPrices = scaler.inverse_transform(model.predict(X))

	st.header(f'Prediction for {ticker}')

	predictedPrice = scaler.inverse_transform(model.predict(X[-1:]))
	formattedPrice = "{:.2f}".format(predictedPrice[0][0])
	st.subheader(f'Predicted Price: {formattedPrice}')

	dates = pd.to_datetime(data['Date'], format='dd-mm-yyyy').iloc[-len(y):].values

	# fig = go.line(scaler.inverse_transform(y))
	# fig.add_scatter(x=predictedPrices, y=predictedPrices, mode='lines')

	fig = plt.figure(figsize=(7, 7))
	plt.plot(dates, scaler.inverse_transform(y), c='g', label='Actual Price')
	plt.plot(dates, predictedPrices, c='r', label='Predicted Price')
	plt.legend()
	plt.show()
	st.pyplot(fig)
else:
	st.write('Model not available')