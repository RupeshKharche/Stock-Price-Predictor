import json

import plotly
import plotly.express as px
import tensorflow as tf
import yfinance as yf
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__, template_folder='templates')

# Min Max Scaler
scaler = MinMaxScaler()

# Loading the model
model = tf.keras.models.load_model("model.h5", compile=False)
model.compile()

# Define the root route
@app.route('/')
def index():
	return render_template('index.html')


@app.route('/callback/<endpoint>')
def cb(endpoint):
	if endpoint == "getStock":
		return gm(request.args.get('data'), request.args.get('period'), request.args.get('interval'))
	elif endpoint == "getInfo":
		stock = request.args.get('data')
		st = yf.Ticker(stock)
		return json.dumps(st.info)
	else:
		return "Bad endpoint", 400


# Return the JSON data for the Plotly graph
def gm(stock, period, interval):
	st = yf.Ticker(stock)

	# Create a line graph
	df = st.history(period=(period), interval=interval)
	df = df.reset_index()
	df.columns = ['Date-Time'] + list(df.columns[1:])
	max = (df['Open'].max())
	min = (df['Open'].min())
	range = max - min
	margin = range * 0.05
	max = max + margin
	min = min - margin
	fig = px.line(df, x='Date-Time', y="Open",
	              hover_data=("Open", "Close", "Volume"),
	              range_y=(min, max), template="seaborn")

	# Create a JSON representation of the graph
	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	# Getting the last 7 close prices of the stock
	windowPrices = df['Close'].to_numpy().reshape(-1, 1)[:-7]

	# Getting the predicted price of the stock
	predictedPrice = predict(windowPrices)

	# Creating a JSON object of the graph and predicted price
	result = json.dumps({
		"graph": graphJSON,
		"predictedPrice": predictedPrice.tolist()
	})

	return result


def predict(windowPrices):
	scaledPrices = scaler.fit_transform(windowPrices)
	predictedPrice = model.predict(scaledPrices)
	return scaler.inverse_transform(predictedPrice)

if __name__ == "__main__":
	app.run(debug=False, port=8001)