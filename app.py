# # from flask import Flask, render_template, request, jsonify
# # import numpy as np
# # import pandas as pd
# # import json
# # import yfinance as yf
# # import plotly.graph_objs as go
# # import plotly.utils
# # from datetime import datetime, timedelta
# # from sklearn.preprocessing import MinMaxScaler
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
# # import xgboost as xgb
# # from sklearn.metrics import mean_squared_error, mean_absolute_error

# # app = Flask(__name__)

# # # ========================== HYBRID MODEL FUNCTIONS ========================== #

# # def fetch_stock_data(ticker):
# #     end = datetime.today().strftime('%Y-%m-%d')
# #     start = (datetime.today() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
# #     data = yf.download(ticker, start=start, end=end)
    
# #     if data.empty:
# #         return None
    
# #     return data[['Close']]

# # def prepare_data(data, time_steps=120):
# #     scaler = MinMaxScaler(feature_range=(0, 1))
# #     scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
# #     X, Y = [], []
# #     for i in range(time_steps, len(scaled_data)):
# #         X.append(scaled_data[i-time_steps:i, 0])
# #         Y.append(scaled_data[i, 0])
# #     return np.array(X), np.array(Y), scaler

# # def train_lstm(X_train, Y_train):
# #     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# #     model = Sequential([
# #         Bidirectional(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1))),
# #         Dropout(0.1),
# #         LSTM(100, return_sequences=False),
# #         Dropout(0.1),
# #         Dense(50, activation="relu"),
# #         Dense(1)
# #     ])
# #     model.compile(optimizer='adam', loss='mean_squared_error')
# #     model.fit(X_train, Y_train, epochs=1, batch_size=16, verbose=0)
# #     return model

# # def train_xgb(X_train, Y_train):
# #     X_train = X_train.reshape(X_train.shape[0], -1)
# #     model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200)
# #     model.fit(X_train, Y_train)
# #     return model

# # def generate_signal(predictions):
# #     if predictions[-1] > predictions[-2]:
# #         return "BUY"
# #     elif predictions[-1] < predictions[-2]:
# #         return "SELL"
# #     else:
# #         return "HOLD"

# # def predict_stock(ticker):
# #     data = fetch_stock_data(ticker)
# #     if data is None:
# #         return None
    
# #     X_lstm, Y_lstm, scaler = prepare_data(data)
# #     X_xgb, Y_xgb = X_lstm, Y_lstm  # Same preprocessing for XGBoost

# #     lstm_model = train_lstm(X_lstm[:-30], Y_lstm[:-30])
# #     xgb_model = train_xgb(X_xgb[:-30], Y_xgb[:-30])
    
# #     future_dates = [datetime.today() + timedelta(days=i) for i in range(1, 31)]

# #     lstm_preds = scaler.inverse_transform(lstm_model.predict(X_lstm[-30:].reshape(-1, 120, 1))).flatten()
# #     xgb_preds = xgb_model.predict(X_xgb[-30:].reshape(30, -1))
# #     final_preds = (lstm_preds + xgb_preds) / 2  

# #     signal = generate_signal(final_preds)
    
# #     actual_prices = scaler.inverse_transform(Y_lstm[-30:].reshape(-1, 1)).flatten()
# #     mse = mean_squared_error(actual_prices, final_preds)
# #     mae = mean_absolute_error(actual_prices, final_preds)
# #     rmse = np.sqrt(mse)
# #     mape = np.mean(np.abs((actual_prices - final_preds) / actual_prices)) * 100

# #     return {
# #         "dates": future_dates,
# #         "actual": actual_prices.tolist(),
# #         "predicted": final_preds.tolist(),
# #         "mse": round(mse, 4),
# #         "mae": round(mae, 4),
# #         "rmse": round(rmse, 4),
# #         "mape": round(mape, 2),
# #         "signal": signal
# #     }

# # # ========================== FLASK ROUTES ========================== #

# # @app.route("/")
# # def home():
# #     return render_template("dashboard.html")

# # @app.route("/predict", methods=["POST"])
# # def predict():
# #     ticker = request.form.get("ticker").upper()
    
# #     results = predict_stock(ticker)
# #     if results is None:
# #         return jsonify({"error": "Invalid Ticker Symbol"}), 400
    
# #     trace_actual = go.Scatter(
# #         x=results["dates"], y=results["actual"], mode="lines", name="Actual Price", line=dict(color="blue")
# #     )
# #     trace_predicted = go.Scatter(
# #         x=results["dates"], y=results["predicted"], mode="lines", name="Predicted Price", line=dict(color="red", dash="dot")
# #     )
    
# #     graph_json = json.dumps({"data": [trace_actual, trace_predicted]}, cls=plotly.utils.PlotlyJSONEncoder)

# #     return render_template("dashboard.html", 
# #                            ticker=ticker, 
# #                            graph_json=graph_json, 
# #                            mse=results["mse"], 
# #                            mae=results["mae"], 
# #                            rmse=results["rmse"], 
# #                            mape=results["mape"], 
# #                            signal=results["signal"])

# # # ========================== DARK/LIGHT THEME ========================== #

# # @app.route("/toggle-theme", methods=["POST"])
# # def toggle_theme():
# #     return jsonify({"status": "Theme switched"})

# # if __name__ == "__main__":
# #     app.run(debug=True)





# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import pandas as pd
# import json
# import yfinance as yf
# import plotly.graph_objs as go
# import plotly.utils
# from datetime import datetime, timedelta
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# app = Flask(__name__)

# # ========================== HYBRID MODEL FUNCTIONS ========================== #

# def fetch_stock_data(ticker):
#     """Fetches historical stock data for the given ticker."""
#     end = datetime.today().strftime('%Y-%m-%d')
#     start = (datetime.today() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
#     data = yf.download(ticker, start=start, end=end)
    
#     if data.empty:
#         return None
    
#     return data[['Close']]

# def prepare_data(data, time_steps=120):
#     """Prepares stock data for training and scaling."""
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
#     X, Y = [], []
#     for i in range(time_steps, len(scaled_data)):
#         X.append(scaled_data[i-time_steps:i, 0])
#         Y.append(scaled_data[i, 0])
#     return np.array(X), np.array(Y), scaler

# def train_lstm(X_train, Y_train):
#     """Trains an LSTM model on the given stock data."""
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     model = Sequential([
#         Bidirectional(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1))),
#         Dropout(0.1),
#         LSTM(100, return_sequences=False),
#         Dropout(0.1),
#         Dense(50, activation="relu"),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(X_train, Y_train, epochs=1, batch_size=16, verbose=0)
#     return model

# def train_xgb(X_train, Y_train):
#     """Trains an XGBoost model on the given stock data."""
#     X_train = X_train.reshape(X_train.shape[0], -1)
#     model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200)
#     model.fit(X_train, Y_train)
#     return model

# def generate_signal(predictions):
#     """Generates Buy/Sell/Hold signals based on stock predictions."""
#     if predictions[-1] > predictions[-2]:
#         return "BUY"
#     elif predictions[-1] < predictions[-2]:
#         return "SELL"
#     else:
#         return "HOLD"

# def predict_stock(ticker):
#     """Fetches data, trains models, and predicts stock prices."""
#     data = fetch_stock_data(ticker)
#     if data is None:
#         return None
    
#     X_lstm, Y_lstm, scaler = prepare_data(data)
#     X_xgb, Y_xgb = X_lstm, Y_lstm  

#     lstm_model = train_lstm(X_lstm[:-30], Y_lstm[:-30])
#     xgb_model = train_xgb(X_xgb[:-30], Y_xgb[:-30])
    
#     future_dates = [datetime.today() + timedelta(days=i) for i in range(1, 31)]

#     lstm_preds = scaler.inverse_transform(lstm_model.predict(X_lstm[-30:].reshape(-1, 120, 1))).flatten()
#     xgb_preds = xgb_model.predict(X_xgb[-30:].reshape(30, -1))
#     final_preds = (lstm_preds + xgb_preds) / 2  

#     signal = generate_signal(final_preds)
    
#     actual_prices = scaler.inverse_transform(Y_lstm[-30:].reshape(-1, 1)).flatten()
#     mse = mean_squared_error(actual_prices, final_preds)
#     mae = mean_absolute_error(actual_prices, final_preds)
#     rmse = np.sqrt(mse)
#     mape = np.mean(np.abs((actual_prices - final_preds) / actual_prices)) * 100

#     return {
#         "dates": future_dates,
#         "actual": actual_prices.tolist(),
#         "predicted": final_preds.tolist(),
#         "mse": round(mse, 4),
#         "mae": round(mae, 4),
#         "rmse": round(rmse, 4),
#         "mape": round(mape, 2),
#         "signal": signal
#     }

# # ========================== FLASK ROUTES ========================== #

# @app.route("/")
# def home():
#     return render_template("dashboard.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     ticker = request.form.get("ticker").upper()
    
#     results = predict_stock(ticker)
#     if results is None:
#         return jsonify({"error": "Invalid Ticker Symbol"}), 400
    
#     trace_actual = go.Scatter(
#         x=results["dates"], y=results["actual"], mode="lines", name="Actual Price", line=dict(color="blue")
#     )
#     trace_predicted = go.Scatter(
#         x=results["dates"], y=results["predicted"], mode="lines", name="Predicted Price", line=dict(color="red", dash="dot")
#     )
    
#     graph_json = json.dumps({"data": [trace_actual, trace_predicted]}, cls=plotly.utils.PlotlyJSONEncoder)

#     return render_template("dashboard.html", 
#                            ticker=ticker, 
#                            graph_json=graph_json, 
#                            mse=results["mse"], 
#                            mae=results["mae"], 
#                            rmse=results["rmse"], 
#                            mape=results["mape"], 
#                            signal=results["signal"])

# if __name__ == "__main__":
#     app.run(debug=True)





import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

# ================== Fetch Stock Data ==================
def fetch_stock_data(ticker):
    end = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end)
    return data[['Close']]

# ================== Prepare Data ==================
def prepare_data(data, time_steps=120):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    X, Y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        Y.append(scaled_data[i, 0])
    return np.array(X), np.array(Y), scaler

# ================== Train LSTM Model ==================
def train_lstm(X_train, Y_train):
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1))),
        Dropout(0.1),
        LSTM(100, return_sequences=False),
        Dropout(0.1),
        Dense(50, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=1, batch_size=16, verbose=0)
    return model

# ================== Train XGBoost Model ==================
def prepare_xgb_data(data, time_steps=120):
    X, Y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i-time_steps:i].values)
        Y.append(data.iloc[i])
    return np.array(X), np.array(Y)

def train_xgb(X_train, Y_train):
    X_train = X_train.reshape(X_train.shape[0], -1)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200)
    model.fit(X_train, Y_train)
    return model

# ================== Generate Buy/Sell Signal ==================
def generate_signal(predictions):
    if predictions[-1] > predictions[-2]:
        return "BUY"
    elif predictions[-1] < predictions[-2]:
        return "SELL"
    else:
        return "HOLD"

# ================== Flask Route: Homepage ==================
@app.route('/')
def homepage():
    return render_template('index.html')  # Home Page

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')  # Dashboard Page


# ================== Flask Route: Stock Prediction ==================
@app.route('/predict', methods=['POST'])
def predict():
    stock_ticker = request.form['ticker']
    data = fetch_stock_data(stock_ticker)
    
    X_lstm, Y_lstm, scaler = prepare_data(data)
    X_xgb, Y_xgb = prepare_xgb_data(data, time_steps=120)
    
    lstm_model = train_lstm(X_lstm[:-30], Y_lstm[:-30])
    xgb_model = train_xgb(X_xgb[:-30], Y_xgb[:-30])
    
    future_dates = [datetime.today() + timedelta(days=i) for i in range(1, 31)]
    
    lstm_preds = scaler.inverse_transform(lstm_model.predict(X_lstm[-30:].reshape(-1, 120, 1))).flatten()
    xgb_preds = xgb_model.predict(X_xgb[-30:].reshape(30, -1))
    final_preds = (lstm_preds + xgb_preds) / 2  
    
    signal = generate_signal(final_preds)
    
    actual_prices = scaler.inverse_transform(Y_lstm[-30:].reshape(-1, 1)).flatten()
    mse = mean_squared_error(actual_prices, final_preds)
    mae = mean_absolute_error(actual_prices, final_preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_prices - final_preds) / actual_prices)) * 100
    
    # Generate Plotly Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=lstm_preds, mode='lines+markers', name='LSTM Prediction', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=future_dates, y=xgb_preds, mode='lines+markers', name='XGBoost Prediction', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=future_dates, y=final_preds, mode='lines+markers', name='Hybrid Prediction', line=dict(color='purple', dash='dot')))
    
    fig.update_layout(title=f'{stock_ticker} Stock Price Prediction for Next 30 Days',
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      legend_title="Prediction Models",
                      template="plotly_white")
    
    graph_html = fig.to_html(full_html=False)
    
    return render_template('dashboard.html', 
                           ticker=stock_ticker,
                           signal=signal, 
                           mse=round(mse, 4), 
                           mae=round(mae, 4),
                           rmse=round(rmse, 4), 
                           mape=round(mape, 2),
                           graph_html=graph_html)

if __name__ == "__main__":
    app.run(debug=True)
