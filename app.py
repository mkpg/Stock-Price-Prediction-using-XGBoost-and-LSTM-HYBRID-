import sys
import re
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
import warnings
import json
import os
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB integration
from db import (
    save_prediction, check_prediction_outcomes, get_analytics_data,
    get_history_data, add_to_watchlist, remove_from_watchlist,
    get_portfolio_data, get_homepage_stats, get_adaptive_params,
    create_user, get_user_by_username, get_user_by_email, get_user_by_id,
    update_user_profile, update_user_password, get_user_stats,
    get_user_by_recovery_phrase
)

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default-dev-key-change-this')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# ============= Config =============

# Token serializer for password reset links
serializer = URLSafeTimedSerializer(app.secret_key)

# ============= Config =============
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")


# ============= Auth Helpers =============
def generate_recovery_phrase():
    """Generate 12 random 4-digit numbers as a recovery mnemonic."""
    return "-".join([str(random.randint(1000, 9999)) for _ in range(12)])


def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_phone(phone):
    """Validate phone number (10-15 digits)."""
    clean = phone.replace(' ', '').replace('-', '').replace('+', '')
    return len(clean) >= 10 and len(clean) <= 15 and clean.isdigit()


def get_logged_in_user():
    """Get the currently logged-in user from session."""
    if 'username' in session:
        return get_user_by_username(session['username'])
    return None


def login_required(f):
    """Decorator to protect routes that require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_user():
    """Make current user available in all templates."""
    user = get_logged_in_user()
    return dict(
        current_user=user,
        logged_in=user is not None
    )


# ================== Fetch Stock Data ==================
def fetch_stock_data(ticker):
    """Fetch historical OHLCV data from Polygon.io — 3 years for relevance"""
    try:
        end = datetime.today().strftime('%Y-%m-%d')
        # Use 3 years: recent data is far more predictive than decade-old patterns
        start = (datetime.today() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
        )
        response = requests.get(url, timeout=15)
        data = response.json()

        if "results" in data and len(data["results"]) > 0:
            df = pd.DataFrame(data["results"])
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            df.set_index("date", inplace=True)
            df = df.rename(columns={"c": "Close", "o": "Open", "h": "High", "l": "Low", "v": "Volume"})
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df = df.dropna()
            return df
        else:
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# ================== Advanced Technical Indicators ==================
def calculate_advanced_indicators(df):
    """Calculate 20+ technical indicators for maximum feature richness"""
    data = df.copy()

    # === Price-based features ===
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

    # Multiple timeframe returns
    for period in [2, 3, 5, 10, 20, 60]:
        data[f'Return_{period}d'] = data['Close'].pct_change(period)

    # === Moving Averages ===
    for period in [5, 10, 20, 50, 100, 200]:
        data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
        data[f'Close_SMA_{period}_ratio'] = data['Close'] / data[f'SMA_{period}']

    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()

    # === RSI (multiple periods) ===
    for period in [7, 14, 21]:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        data[f'RSI_{period}'] = 100 - (100 / (1 + rs))

    # === MACD ===
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

    # === Bollinger Bands ===
    bb_sma = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = bb_sma + (bb_std * 2)
    data['BB_Lower'] = bb_sma - (bb_std * 2)
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / bb_sma
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'] + 1e-10)

    # === ATR (Average True Range) ===
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR_14'] = tr.rolling(window=14).mean()
    data['ATR_Ratio'] = data['ATR_14'] / data['Close']

    # === Stochastic Oscillator ===
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['Stoch_K'] = 100 * (data['Close'] - low_14) / (high_14 - low_14 + 1e-10)
    data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()

    # === Williams %R ===
    data['Williams_R'] = -100 * (high_14 - data['Close']) / (high_14 - low_14 + 1e-10)

    # === OBV (On Balance Volume) ===
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv
    data['OBV_SMA'] = data['OBV'].rolling(window=20).mean()

    # === Volume Features ===
    data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / (data['Volume_SMA_20'] + 1e-10)
    data['Volume_Change'] = data['Volume'].pct_change()

    # === Volatility ===
    data['Volatility_10'] = data['Returns'].rolling(window=10).std()
    data['Volatility_20'] = data['Returns'].rolling(window=20).std()
    data['Volatility_60'] = data['Returns'].rolling(window=60).std()

    # === Price Momentum ===
    data['Momentum_10'] = data['Close'] - data['Close'].shift(10)
    data['Momentum_20'] = data['Close'] - data['Close'].shift(20)
    data['ROC_10'] = ((data['Close'] - data['Close'].shift(10)) / (data['Close'].shift(10) + 1e-10)) * 100

    # === Candlestick Patterns ===
    data['Body_Size'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low'] + 1e-10)
    data['Upper_Shadow'] = (data['High'] - data[['Close', 'Open']].max(axis=1)) / (data['High'] - data['Low'] + 1e-10)
    data['Lower_Shadow'] = (data[['Close', 'Open']].min(axis=1) - data['Low']) / (data['High'] - data['Low'] + 1e-10)

    # === Mean Reversion ===
    data['Distance_SMA_20'] = (data['Close'] - data['SMA_20']) / (data['SMA_20'] + 1e-10)
    data['Distance_SMA_50'] = (data['Close'] - data['SMA_50']) / (data['SMA_50'] + 1e-10)

    # === Trend Strength ===
    data['ADX_proxy'] = abs(data['Return_5d']) / (data['Volatility_10'] + 1e-10)

    # Drop rows with NaN from indicators
    data = data.dropna()

    return data


# ================== Feature Selection ==================
def select_features(data):
    """Select the most predictive features for the models"""
    feature_cols = [
        'Returns', 'Return_2d', 'Return_3d', 'Return_5d', 'Return_10d', 'Return_20d',
        'Close_SMA_5_ratio', 'Close_SMA_10_ratio', 'Close_SMA_20_ratio',
        'Close_SMA_50_ratio', 'Close_SMA_100_ratio',
        'RSI_7', 'RSI_14', 'RSI_21',
        'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Width', 'BB_Position',
        'ATR_Ratio',
        'Stoch_K', 'Stoch_D', 'Williams_R',
        'Volume_Ratio', 'Volume_Change',
        'Volatility_10', 'Volatility_20',
        'Momentum_10', 'ROC_10',
        'Body_Size', 'Upper_Shadow', 'Lower_Shadow',
        'Distance_SMA_20', 'Distance_SMA_50',
        'ADX_proxy'
    ]
    # Only keep features that exist
    available = [c for c in feature_cols if c in data.columns]
    return available


# ================== Prepare LSTM Data ==================
def prepare_lstm_data(data, feature_cols, time_steps=30):
    """Prepare sequential data for LSTM with PROPER target scaling"""
    feature_data = data[feature_cols].values
    close_prices = data['Close'].values

    # Scale features
    feat_scaler = RobustScaler()
    scaled_features = feat_scaler.fit_transform(feature_data)

    # CRITICAL: Scale targets too — LSTM can't learn raw $350 targets well
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled = close_scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()

    X, Y = [], []
    for i in range(time_steps, len(scaled_features)):
        X.append(scaled_features[i-time_steps:i])
        Y.append(close_scaled[i])  # Scaled target!

    return np.array(X), np.array(Y), feat_scaler, close_scaler


# ================== Prepare XGBoost Data ==================
def prepare_xgb_data(data, feature_cols, time_steps=30):
    """Prepare flattened + statistical features for XGBoost"""
    feature_data = data[feature_cols].values
    close_prices = data['Close'].values

    # Scale close prices for XGBoost targets too
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled = close_scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()

    X, Y = [], []
    for i in range(time_steps, len(feature_data)):
        window = feature_data[i-time_steps:i]

        # Flatten recent window (last 5 days)
        recent = window[-5:].flatten()

        # Statistical aggregates over full window
        stats = []
        for col_idx in range(window.shape[1]):
            col_data = window[:, col_idx]
            stats.extend([
                np.mean(col_data),
                np.std(col_data),
                np.min(col_data),
                np.max(col_data),
                col_data[-1] - col_data[0],
            ])

        # Close price features (use raw for XGB features, but scaled for targets)
        close_window = close_prices[i-time_steps:i]
        close_stats = [
            np.mean(close_window),
            np.std(close_window),
            (close_window[-1] - close_window[0]) / (close_window[0] + 1e-10),
            np.max(close_window) - np.min(close_window),
        ]

        features = np.concatenate([recent, stats, close_stats])
        X.append(features)
        Y.append(close_scaled[i])  # Scaled target!

    return np.array(X), np.array(Y), close_scaler


# ================== Train LSTM Model ==================
def train_lstm_model(X_train, Y_train, X_val, Y_val):
    """Train LSTM with advanced architecture and proper validation"""
    n_features = X_train.shape[2]
    n_steps = X_train.shape[1]

    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)),
                      input_shape=(n_steps, n_features)),
        LayerNormalization(),
        Dropout(0.3),

        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))),
        LayerNormalization(),
        Dropout(0.3),

        LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001)),
        Dropout(0.2),

        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.0001),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0)
    ]

    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
        shuffle=False  # Important for time series
    )

    return model


# ================== Train XGBoost Model ==================
def train_xgb_model(X_train, Y_train, X_val, Y_val):
    """Train XGBoost with optimized hyperparameters and early stopping"""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, Y_train,
        eval_set=[(X_val, Y_val)],
        verbose=False
    )

    return model


# ================== Directional Accuracy ==================
def calculate_directional_accuracy(actual, predicted):
    """Calculate percentage of correct up/down movement predictions"""
    if len(actual) < 2:
        return 0.0
    actual_dir = np.diff(actual) > 0
    pred_dir = np.diff(predicted) > 0
    accuracy = np.mean(actual_dir == pred_dir) * 100
    return accuracy


# ================== Walk-Forward Validation ==================
def walk_forward_predict(data, feature_cols, time_steps=30):
    """
    Walk-forward validation with PROPER scaling.
    All predictions are in scaled space, then inverse-transformed for metrics.
    """
    n = len(data)
    close_prices_raw = data['Close'].values  # Keep raw prices for inverse transform

    # Prepare all data (targets are NOW scaled 0-1)
    X_lstm_all, Y_lstm_all, feat_scaler, lstm_close_scaler = prepare_lstm_data(data, feature_cols, time_steps)
    X_xgb_all, Y_xgb_all, xgb_close_scaler = prepare_xgb_data(data, feature_cols, time_steps)

    # Split: 70% train, 15% validation, 15% test
    total_samples = len(X_lstm_all)
    val_size = min(60, max(20, int(total_samples * 0.15)))
    test_size = min(60, max(20, int(total_samples * 0.15)))

    train_end = total_samples - val_size - test_size
    val_end = total_samples - test_size

    # Ensure minimum training data
    if train_end < 100:
        train_end = max(50, total_samples - 40)
        val_size = (total_samples - train_end) // 2
        val_end = train_end + val_size
        test_size = total_samples - val_end

    X_train_lstm = X_lstm_all[:train_end]
    Y_train_lstm = Y_lstm_all[:train_end]
    X_val_lstm = X_lstm_all[train_end:val_end]
    Y_val_lstm = Y_lstm_all[train_end:val_end]
    X_test_lstm = X_lstm_all[val_end:]
    Y_test_lstm = Y_lstm_all[val_end:]

    X_train_xgb = X_xgb_all[:train_end]
    Y_train_xgb = Y_xgb_all[:train_end]
    X_val_xgb = X_xgb_all[train_end:val_end]
    Y_val_xgb = Y_xgb_all[train_end:val_end]
    X_test_xgb = X_xgb_all[val_end:]
    Y_test_xgb = Y_xgb_all[val_end:]

    # Train models (on scaled targets)
    lstm_model = train_lstm_model(X_train_lstm, Y_train_lstm, X_val_lstm, Y_val_lstm)
    xgb_model = train_xgb_model(X_train_xgb, Y_train_xgb, X_val_xgb, Y_val_xgb)

    # --- Validation predictions (scaled space) ---
    lstm_val_pred_scaled = lstm_model.predict(X_val_lstm, verbose=0).flatten()
    xgb_val_pred_scaled = xgb_model.predict(X_val_xgb)

    # Inverse transform validation predictions to real prices for accuracy
    lstm_val_real = lstm_close_scaler.inverse_transform(lstm_val_pred_scaled.reshape(-1, 1)).flatten()
    xgb_val_real = xgb_close_scaler.inverse_transform(xgb_val_pred_scaled.reshape(-1, 1)).flatten()
    val_actual_real = lstm_close_scaler.inverse_transform(Y_val_lstm.reshape(-1, 1)).flatten()

    # Directional accuracy on real prices
    lstm_val_acc = calculate_directional_accuracy(val_actual_real, lstm_val_real)
    xgb_val_acc = calculate_directional_accuracy(val_actual_real, xgb_val_real)

    # Dynamic ensemble weights
    total_acc = lstm_val_acc + xgb_val_acc + 1e-10
    lstm_weight = lstm_val_acc / total_acc
    xgb_weight = xgb_val_acc / total_acc

    # --- Test predictions (scaled space) ---
    lstm_test_pred_scaled = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    xgb_test_pred_scaled = xgb_model.predict(X_test_xgb)

    # Inverse transform to real prices
    lstm_test_real = lstm_close_scaler.inverse_transform(lstm_test_pred_scaled.reshape(-1, 1)).flatten()
    xgb_test_real = xgb_close_scaler.inverse_transform(xgb_test_pred_scaled.reshape(-1, 1)).flatten()
    test_actual_real = lstm_close_scaler.inverse_transform(Y_test_lstm.reshape(-1, 1)).flatten()

    # Ensemble in real price space
    ensemble_test_real = lstm_weight * lstm_test_real + xgb_weight * xgb_test_real

    # --- Bias correction using linear fit (much better than simple ratio) ---
    try:
        from numpy.polynomial import polynomial as P
        # Fit a linear correction: actual = a * predicted + b
        coeffs = np.polyfit(ensemble_test_real, test_actual_real, 1)
        ensemble_test_corrected = np.polyval(coeffs, ensemble_test_real)
        # Only use correction if it improves things
        r2_before = r2_score(test_actual_real, ensemble_test_real)
        r2_after = r2_score(test_actual_real, ensemble_test_corrected)
        if r2_after > r2_before:
            ensemble_test_real = ensemble_test_corrected
            bias_coeffs = coeffs
        else:
            bias_coeffs = None
    except Exception:
        bias_coeffs = None

    # --- Metrics (all in real price space) ---
    dir_accuracy = calculate_directional_accuracy(test_actual_real, ensemble_test_real)
    lstm_dir_acc = calculate_directional_accuracy(test_actual_real, lstm_test_real)
    xgb_dir_acc = calculate_directional_accuracy(test_actual_real, xgb_test_real)

    mse = mean_squared_error(test_actual_real, ensemble_test_real)
    mae_val = mean_absolute_error(test_actual_real, ensemble_test_real)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test_actual_real - ensemble_test_real) / (test_actual_real + 1e-10))) * 100
    r2 = r2_score(test_actual_real, ensemble_test_real)

    # --- Future predictions ---
    # Use last 30 windows for future forecast
    n_future = min(30, len(X_lstm_all))
    future_lstm_scaled = lstm_model.predict(X_lstm_all[-n_future:], verbose=0).flatten()
    future_xgb_scaled = xgb_model.predict(X_xgb_all[-n_future:])

    future_lstm_real = lstm_close_scaler.inverse_transform(future_lstm_scaled.reshape(-1, 1)).flatten()
    future_xgb_real = xgb_close_scaler.inverse_transform(future_xgb_scaled.reshape(-1, 1)).flatten()
    future_ensemble_real = lstm_weight * future_lstm_real + xgb_weight * future_xgb_real

    # Apply bias correction to future predictions
    if bias_coeffs is not None:
        future_ensemble_real = np.polyval(bias_coeffs, future_ensemble_real)

    # Anchor future predictions to the last known price for realism
    last_price = close_prices_raw[-1]
    future_mean = np.mean(future_ensemble_real)
    if future_mean > 0:
        anchor_ratio = last_price / future_mean
        future_ensemble_real = future_ensemble_real * anchor_ratio
        future_lstm_real = future_lstm_real * anchor_ratio
        future_xgb_real = future_xgb_real * anchor_ratio

    return {
        'lstm_model': lstm_model,
        'xgb_model': xgb_model,
        'test_actual': test_actual_real,
        'test_dates': data.index[val_end + time_steps:val_end + time_steps + len(test_actual_real)],
        'ensemble_pred': ensemble_test_real,
        'lstm_pred': lstm_test_real,
        'xgb_pred': xgb_test_real,
        'future_lstm': future_lstm_real,
        'future_xgb': future_xgb_real,
        'future_ensemble': future_ensemble_real,
        'dir_accuracy': dir_accuracy,
        'lstm_accuracy': lstm_dir_acc,
        'xgb_accuracy': xgb_dir_acc,
        'lstm_weight': lstm_weight,
        'xgb_weight': xgb_weight,
        'mse': mse,
        'mae': mae_val,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
    }


# ================== Signal Generation ==================
def generate_signal(predictions, dir_accuracy, adaptive_params=None):
    """Generate trading signal with confidence, adjusted by adaptive learning"""
    if len(predictions) < 5:
        return "HOLD", "Low"

    # Short-term trend (last 5 predictions)
    short_trend = np.polyfit(range(5), predictions[-5:], 1)[0]
    price_change_pct = ((predictions[-1] - predictions[-5]) / (predictions[-5] + 1e-10)) * 100

    # Medium-term trend
    if len(predictions) >= 10:
        med_trend = np.polyfit(range(10), predictions[-10:], 1)[0]
    else:
        med_trend = short_trend

    # Consistency check
    recent_dirs = np.diff(predictions[-5:]) > 0
    consistency = np.mean(recent_dirs)

    # Base confidence level from directional accuracy
    if dir_accuracy >= 80:
        confidence = "Very High"
    elif dir_accuracy >= 70:
        confidence = "High"
    elif dir_accuracy >= 60:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Signal logic
    if short_trend > 0 and med_trend > 0 and price_change_pct > 0.5 and consistency >= 0.6:
        signal = "STRONG BUY"
    elif short_trend > 0 and price_change_pct > 0.2:
        signal = "BUY"
    elif short_trend < 0 and med_trend < 0 and price_change_pct < -0.5 and consistency <= 0.4:
        signal = "STRONG SELL"
    elif short_trend < 0 and price_change_pct < -0.2:
        signal = "SELL"
    else:
        signal = "HOLD"

    # ===== ADAPTIVE ADJUSTMENT =====
    if adaptive_params and adaptive_params.get("has_history"):
        conf_mod = adaptive_params.get("confidence_modifier", 1.0)
        sig_rel = adaptive_params.get("signal_reliability", {})

        # Adjust confidence based on historical win rate
        if conf_mod < 0.7:
            # Downgrade confidence if model historically bad on this ticker
            if confidence == "Very High":
                confidence = "High"
            elif confidence == "High":
                confidence = "Medium"
            elif confidence == "Medium":
                confidence = "Low"
        elif conf_mod > 1.2:
            # Upgrade confidence if model historically good
            if confidence == "Low":
                confidence = "Medium"
            elif confidence == "Medium":
                confidence = "High"

        # If this specific signal type is unreliable, add caution
        signal_reliability = sig_rel.get(signal, 1.0)
        if signal_reliability < 0.6 and signal in ["STRONG BUY", "STRONG SELL"]:
            # Downgrade strong signals to regular if historically they've been wrong
            if signal == "STRONG BUY":
                signal = "BUY"
            elif signal == "STRONG SELL":
                signal = "SELL"
            confidence = "Low"

    return signal, confidence


# ================== Auth Routes ==================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page."""
    if 'username' in session:
        return redirect(url_for('homepage'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = get_user_by_username(username)

        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            session.permanent = True
            flash(f'Welcome back, {user["first_name"]}!', 'success')
            return redirect(url_for('homepage'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page."""
    if 'username' in session:
        return redirect(url_for('homepage'))

    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        phone = request.form.get('phone', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        position_size = request.form.get('position_size', '10000')

        errors = []

        if not all([first_name, last_name, username, email, phone, password]):
            errors.append('All fields are required.')

        if len(password) < 6:
            errors.append('Password must be at least 6 characters.')

        if password != confirm_password:
            errors.append('Passwords do not match.')

        if not validate_email(email):
            errors.append('Invalid email format.')

        if not validate_phone(phone):
            errors.append('Invalid phone number format.')

        if get_user_by_username(username):
            errors.append('Username already exists.')

        if get_user_by_email(email):
            errors.append('Email already registered.')

        try:
            position_size = float(position_size)
            if position_size < 100 or position_size > 10000000:
                errors.append('Position size must be between $100 and $10,000,000.')
        except ValueError:
            position_size = 10000

        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('register.html')

        hashed_pw = generate_password_hash(password)
        recovery_phrase = generate_recovery_phrase()
        
        try:
            create_user(first_name, last_name, username, email, phone, hashed_pw, recovery_phrase, position_size)
            # Store the phrase in session temporarily so we can show it on a success page
            session['new_recovery_phrase'] = recovery_phrase
            return render_template('registration_success.html', phrase=recovery_phrase, phrase_list=recovery_phrase.split('-'))
        except Exception as e:
            flash('Registration failed. Please try again.', 'danger')
            print(f"Registration error: {e}")

    return render_template('register.html')


@app.route('/logout')
def logout():
    """Logout and clear session."""
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/profile')
@login_required
def profile():
    """View user profile with stats."""
    user = get_logged_in_user()
    stats = get_user_stats(user['username'])
    return render_template('profile.html', user=user, stats=stats)


@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Edit user profile."""
    user = get_logged_in_user()

    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        phone = request.form.get('phone', '').strip()
        position_size = request.form.get('position_size', '10000')

        errors = []

        if not validate_email(email):
            errors.append('Invalid email format.')

        if not validate_phone(phone):
            errors.append('Invalid phone number format.')

        # Check email uniqueness (excluding self)
        existing = get_user_by_email(email)
        if existing and existing['username'] != user['username']:
            errors.append('Email is already in use by another account.')

        try:
            position_size = float(position_size)
            if position_size < 100 or position_size > 10000000:
                errors.append('Position size must be between $100 and $10,000,000.')
        except ValueError:
            position_size = user.get('position_size', 10000)

        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('edit_profile.html', user=user)

        update_user_profile(user['username'], {
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'phone': phone,
            'position_size': position_size,
        })

        flash('Profile updated successfully.', 'success')
        return redirect(url_for('profile'))

    return render_template('edit_profile.html', user=user)


@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change password while logged in."""
    user = get_logged_in_user()

    if request.method == 'POST':
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not check_password_hash(user['password'], current_password):
            flash('Current password is incorrect.', 'danger')
            return render_template('change_password.html')

        if new_password != confirm_password:
            flash('New passwords do not match.', 'danger')
            return render_template('change_password.html')

        if len(new_password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('change_password.html')

        hashed_pw = generate_password_hash(new_password)
        update_user_password(user['username'], hashed_pw)

        flash('Password changed successfully.', 'success')
        return redirect(url_for('profile'))

    return render_template('change_password.html')


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    """Password reset request via 12-number recovery phrase."""
    if request.method == 'POST':
        # Collect all 12 numbers from form
        phrase_parts = [request.form.get(f'n{i}', '').strip() for i in range(1, 13)]
        full_phrase = "-".join(phrase_parts)
        
        user = get_user_by_recovery_phrase(full_phrase)

        if user:
            # If phrase matches, put user ID in session temporarily for the reset page
            session['reset_user_id'] = user['username']
            flash('Identity verified! Please set your new password.', 'success')
            return redirect(url_for('reset_password'))
        
        flash('Invalid recovery phrase. Please check the numbers and try again.', 'danger')

    return render_template('forgot_password.html')


@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    """Reset password page after mnemonic verification."""
    username = session.get('reset_user_id')
    if not username:
        flash('Session expired or unauthorized access.', 'danger')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('reset_password.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('reset_password.html')

        hashed_pw = generate_password_hash(password)
        update_user_password(username, hashed_pw)
        
        # Clear the reset session
        session.pop('reset_user_id', None)
        
        flash('Password reset successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html')


# ================== Flask Routes ==================

@app.route('/')
def homepage():
    """Homepage with LIVE stats from MongoDB"""
    stats = get_homepage_stats()
    return render_template('index.html', stats=stats)


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        stock_ticker = request.form.get('ticker', '').upper().strip()

        if not stock_ticker:
            return render_template('index.html', error="Please enter a valid ticker symbol.", stats=get_homepage_stats())

        # Fetch data
        data = fetch_stock_data(stock_ticker)
        if data is None or len(data) < 200:
            return render_template('index.html',
                                   ticker=stock_ticker,
                                   error=f"Insufficient data for {stock_ticker}. Need 200+ days of history.",
                                   stats=get_homepage_stats())

        # Calculate indicators
        data_enriched = calculate_advanced_indicators(data)

        if len(data_enriched) < 200:
            return render_template('index.html',
                                   ticker=stock_ticker,
                                   error="Not enough processed data after indicator calculation.",
                                   stats=get_homepage_stats())

        # Select features
        feature_cols = select_features(data_enriched)

        # ===== ADAPTIVE LEARNING: Get historical performance for this ticker =====
        adaptive_params = get_adaptive_params(stock_ticker)
        print(f"  [Adaptive] {stock_ticker}: {adaptive_params.get('learning_note', 'No history')}")

        # Run prediction pipeline
        results = walk_forward_predict(data_enriched, feature_cols, time_steps=30)

        # ===== Apply adaptive weight adjustments if we have history =====
        if adaptive_params.get("has_history"):
            # Blend model weights with historical performance
            hist_lstm_w = adaptive_params["lstm_weight_adj"]
            hist_xgb_w = adaptive_params["xgb_weight_adj"]
            cur_lstm_w = results["lstm_weight"]
            cur_xgb_w = results["xgb_weight"]

            # 40% historical, 60% current run (trust current data more)
            blend = 0.4
            adj_lstm_w = blend * hist_lstm_w + (1 - blend) * cur_lstm_w
            adj_xgb_w = blend * hist_xgb_w + (1 - blend) * cur_xgb_w

            # Normalize
            total_w = adj_lstm_w + adj_xgb_w + 1e-10
            adj_lstm_w = adj_lstm_w / total_w
            adj_xgb_w = adj_xgb_w / total_w

            # Recalculate future ensemble with adjusted weights
            results['future_ensemble'] = (
                adj_lstm_w * results['future_lstm'] +
                adj_xgb_w * results['future_xgb']
            )
            results['lstm_weight'] = adj_lstm_w
            results['xgb_weight'] = adj_xgb_w
            print(f"  [Adaptive] Adjusted weights: LSTM={adj_lstm_w:.1%}, XGB={adj_xgb_w:.1%}")

        # Generate signal (with adaptive adjustment)
        signal, confidence = generate_signal(
            results['future_ensemble'], results['dir_accuracy'], adaptive_params
        )

        # Get current price (last close)
        current_price = float(data['Close'].iloc[-1])

        # Determine predicted direction
        future_prices = results['future_ensemble'].tolist()
        predicted_direction = "up" if future_prices[-1] > current_price else "down"

        # Determine signal class for UI
        signal_class = 'buy' if 'BUY' in signal else ('sell' if 'SELL' in signal else 'hold')

        # Build learning info for the UI
        learning_info = {
            "has_history": adaptive_params.get("has_history", False),
            "note": adaptive_params.get("learning_note", ""),
            "win_rate": adaptive_params.get("overall_win_rate", 0),
            "total_outcomes": adaptive_params.get("total_outcomes", 0),
            "best_model": adaptive_params.get("best_model", "Ensemble"),
        }

        # ===== SAVE TO MONGODB =====
        prediction_record = {
            "ticker": stock_ticker,
            "signal": signal,
            "signal_class": signal_class,
            "confidence": confidence,
            "directional_accuracy": round(results['dir_accuracy'], 1),
            "mape": round(results['mape'], 2),
            "r2": round(results['r2'], 4),
            "mse": round(results['mse'], 4),
            "mae": round(results['mae'], 4),
            "rmse": round(results['rmse'], 4),
            "lstm_accuracy": round(results['lstm_accuracy'], 1),
            "xgb_accuracy": round(results['xgb_accuracy'], 1),
            "lstm_weight": round(results['lstm_weight'] * 100, 1),
            "xgb_weight": round(results['xgb_weight'] * 100, 1),
            "num_features": len(feature_cols),
            "data_points": len(data_enriched),
            "current_price": current_price,
            "predicted_prices": future_prices,
            "predicted_direction": predicted_direction,
            "user_id": session.get('username'),
            "position_size": get_logged_in_user().get('position_size', 10000) if get_logged_in_user() else 10000,
        }
        save_prediction(prediction_record)

        # Also check pending outcomes in background (triggers adaptive learning)
        try:
            check_prediction_outcomes(POLYGON_API_KEY)
        except Exception:
            pass

        # Future dates (trading days only)
        future_dates = []
        current_date = datetime.today()
        days_added = 0
        while days_added < 30:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:
                future_dates.append(current_date)
                days_added += 1

        # Create visualization
        fig = go.Figure()

        # Test actual vs predicted
        test_dates_list = list(results['test_dates'])
        if len(test_dates_list) > 0:
            fig.add_trace(go.Scatter(
                x=test_dates_list, y=results['test_actual'],
                mode='lines', name='Actual Price',
                line=dict(color='#00d4ff', width=2.5)
            ))
            fig.add_trace(go.Scatter(
                x=test_dates_list, y=results['ensemble_pred'],
                mode='lines', name='Ensemble Prediction',
                line=dict(color='#ff6b35', width=2, dash='dash')
            ))

        # Future predictions
        fig.add_trace(go.Scatter(
            x=future_dates, y=results['future_lstm'],
            mode='lines+markers', name=f'LSTM ({results["lstm_accuracy"]:.1f}% acc)',
            line=dict(color='#00ff88', width=1.5),
            marker=dict(size=4)
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=results['future_xgb'],
            mode='lines+markers', name=f'XGBoost ({results["xgb_accuracy"]:.1f}% acc)',
            line=dict(color='#ff4757', width=1.5),
            marker=dict(size=4)
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=results['future_ensemble'],
            mode='lines+markers', name=f'Ensemble ({results["dir_accuracy"]:.1f}% acc)',
            line=dict(color='#ffd700', width=3),
            marker=dict(size=6, symbol='diamond')
        ))

        fig.update_layout(
            title=dict(
                text=f'{stock_ticker} — AI Prediction Engine',
                font=dict(size=20, color='#e4e4e7')
            ),
            xaxis=dict(
                title='Date',
                gridcolor='rgba(255,255,255,0.05)',
                color='#a1a1a6'
            ),
            yaxis=dict(
                title='Price ($)',
                gridcolor='rgba(255,255,255,0.05)',
                color='#a1a1a6'
            ),
            hovermode='x unified',
            template='plotly_dark',
            plot_bgcolor='rgba(15, 15, 30, 0.8)',
            paper_bgcolor='rgba(15, 15, 30, 0)',
            height=550,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='#a1a1a6')
            ),
            margin=dict(l=60, r=30, t=80, b=50)
        )

        graph_html = fig.to_html(full_html=False, config={'displayModeBar': True, 'scrollZoom': True})

        return render_template('index.html',
                               ticker=stock_ticker,
                               signal=signal,
                               signal_class=signal_class,
                               confidence=confidence,
                               directional_accuracy=round(results['dir_accuracy'], 1),
                               lstm_accuracy=round(results['lstm_accuracy'], 1),
                               xgb_accuracy=round(results['xgb_accuracy'], 1),
                               lstm_weight=round(results['lstm_weight'] * 100, 1),
                               xgb_weight=round(results['xgb_weight'] * 100, 1),
                               mse=round(results['mse'], 4),
                               mae=round(results['mae'], 4),
                               rmse=round(results['rmse'], 4),
                               mape=round(results['mape'], 2),
                               r2=round(results['r2'], 4),
                               graph_html=graph_html,
                               num_features=len(feature_cols),
                               data_points=len(data_enriched),
                               now=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               learning_info=learning_info,
                               stats=get_homepage_stats())

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"Prediction error: {str(e)}", stats=get_homepage_stats())


# ================== API Endpoint for AJAX predictions ==================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for async predictions"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper().strip()

        if not ticker:
            return jsonify({'error': 'No ticker provided'}), 400

        stock_data = fetch_stock_data(ticker)
        if stock_data is None or len(stock_data) < 200:
            return jsonify({'error': f'Insufficient data for {ticker}'}), 400

        enriched = calculate_advanced_indicators(stock_data)
        features = select_features(enriched)

        # Adaptive learning
        adaptive_params = get_adaptive_params(ticker)

        results = walk_forward_predict(enriched, features, time_steps=30)

        # Apply adaptive weight adjustments
        if adaptive_params.get("has_history"):
            hist_lstm_w = adaptive_params["lstm_weight_adj"]
            hist_xgb_w = adaptive_params["xgb_weight_adj"]
            cur_lstm_w = results["lstm_weight"]
            cur_xgb_w = results["xgb_weight"]
            blend = 0.4
            adj_lstm_w = blend * hist_lstm_w + (1 - blend) * cur_lstm_w
            adj_xgb_w = blend * hist_xgb_w + (1 - blend) * cur_xgb_w
            total_w = adj_lstm_w + adj_xgb_w + 1e-10
            adj_lstm_w /= total_w
            adj_xgb_w /= total_w
            results['future_ensemble'] = (
                adj_lstm_w * results['future_lstm'] +
                adj_xgb_w * results['future_xgb']
            )
            results['lstm_weight'] = adj_lstm_w
            results['xgb_weight'] = adj_xgb_w

        signal, confidence = generate_signal(
            results['future_ensemble'], results['dir_accuracy'], adaptive_params
        )

        current_price = float(stock_data['Close'].iloc[-1])
        future_prices = results['future_ensemble'].tolist()
        predicted_direction = "up" if future_prices[-1] > current_price else "down"
        signal_class = 'buy' if 'BUY' in signal else ('sell' if 'SELL' in signal else 'hold')

        # Save to MongoDB
        prediction_record = {
            "ticker": ticker,
            "signal": signal,
            "signal_class": signal_class,
            "confidence": confidence,
            "directional_accuracy": round(results['dir_accuracy'], 1),
            "mape": round(results['mape'], 2),
            "r2": round(results['r2'], 4),
            "mse": round(results['mse'], 4),
            "mae": round(results['mae'], 4),
            "rmse": round(results['rmse'], 4),
            "lstm_accuracy": round(results['lstm_accuracy'], 1),
            "xgb_accuracy": round(results['xgb_accuracy'], 1),
            "lstm_weight": round(results['lstm_weight'] * 100, 1),
            "xgb_weight": round(results['xgb_weight'] * 100, 1),
            "num_features": len(features),
            "data_points": len(enriched),
            "current_price": current_price,
            "predicted_prices": future_prices,
            "predicted_direction": predicted_direction,
            "user_id": session.get('username'),
            "position_size": get_logged_in_user().get('position_size', 10000) if get_logged_in_user() else 10000,
        }
        save_prediction(prediction_record)

        # Trigger outcome checking (which triggers adaptive learning)
        try:
            check_prediction_outcomes(POLYGON_API_KEY)
        except Exception:
            pass

        return jsonify({
            'ticker': ticker,
            'signal': signal,
            'confidence': confidence,
            'accuracy': round(results['dir_accuracy'], 1),
            'lstm_accuracy': round(results['lstm_accuracy'], 1),
            'xgb_accuracy': round(results['xgb_accuracy'], 1),
            'mae': round(results['mae'], 4),
            'mape': round(results['mape'], 2),
            'r2': round(results['r2'], 4),
            'future_prices': future_prices,
            'adaptive_learning': {
                'active': adaptive_params.get('has_history', False),
                'win_rate': adaptive_params.get('overall_win_rate', 0),
                'note': adaptive_params.get('learning_note', ''),
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ================== Analytics — LIVE DATA ==================
@app.route('/analytics')
@login_required
def analytics():
    """Analytics page with ALL data from MongoDB"""
    # Check outcomes for pending predictions
    try:
        check_prediction_outcomes(POLYGON_API_KEY)
    except Exception:
        pass

    data = get_analytics_data(user_id=session.get('username'))
    return render_template('analytics.html', data=data)


# ================== Portfolio — LIVE DATA ==================
@app.route('/portfolio')
@login_required
def portfolio():
    """Portfolio page with live prices and real watchlist"""
    portfolio_data = get_portfolio_data(session.get('username'), POLYGON_API_KEY)
    return render_template('portfolio.html', data=portfolio_data)


@app.route('/api/watchlist/add', methods=['POST'])
@login_required
def api_add_watchlist():
    """Add a stock to the watchlist"""
    req = request.get_json()
    ticker = req.get('ticker', '').upper().strip()
    entry_price_str = req.get('entry_price', '0')

    if not ticker:
        return jsonify({'error': 'No ticker provided'}), 400

    try:
        entry_price = float(entry_price_str) if entry_price_str else 0
    except (ValueError, TypeError):
        entry_price = 0

    user_id = session.get('username')
    result = add_to_watchlist(user_id, ticker, entry_price, POLYGON_API_KEY)
    if result:
        return jsonify({'success': True, 'ticker': ticker, 'entry_price': result['entry_price']})
    else:
        return jsonify({'error': f'Could not add {ticker}. Check if the ticker is valid.'}), 400


@app.route('/api/watchlist/remove', methods=['POST'])
@login_required
def api_remove_watchlist():
    """Remove a stock from the watchlist"""
    req = request.get_json()
    ticker = req.get('ticker', '').upper().strip()

    if not ticker:
        return jsonify({'error': 'No ticker provided'}), 400

    user_id = session.get('username')
    success = remove_from_watchlist(user_id, ticker)
    if success:
        return jsonify({'success': True, 'ticker': ticker})
    else:
        return jsonify({'error': f'{ticker} not found in watchlist'}), 404


# ================== History — LIVE DATA ==================
@app.route('/history')
@login_required
def history():

    """History page with real paginated data from MongoDB"""
    # Check outcomes for pending predictions
    try:
        check_prediction_outcomes(POLYGON_API_KEY)
    except Exception:
        pass

    page = request.args.get('page', 1, type=int)
    signal_filter = request.args.get('signal', 'all')
    min_accuracy = request.args.get('min_accuracy', None)
    outcome_filter = request.args.get('outcome', 'all')

    # Get history with filters
    history_data = get_history_data(
        user_id=session.get('username'),
        page=page,
        per_page=15,
        signal_filter=signal_filter,
        min_accuracy=min_accuracy,
        outcome_filter=outcome_filter
    )

    return render_template('history.html',
                           data=history_data,
                           current_signal=signal_filter,
                           current_accuracy=min_accuracy or '',
                           current_outcome=outcome_filter)


# ================== Quick Quote API ==================
@app.route('/api/quote/<ticker>')
def quick_quote(ticker):
    """Get current price for a ticker"""
    try:
        ticker = ticker.upper()
        end = datetime.today().strftime('%Y-%m-%d')
        start = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start}/{end}?adjusted=true&sort=desc&limit=1&apiKey={POLYGON_API_KEY}"
        )
        resp = requests.get(url, timeout=10).json()
        if 'results' in resp and len(resp['results']) > 0:
            r = resp['results'][0]
            return jsonify({
                'ticker': ticker,
                'price': r['c'],
                'open': r['o'],
                'high': r['h'],
                'low': r['l'],
                'volume': r['v'],
                'change': round(((r['c'] - r['o']) / r['o']) * 100, 2)
            })
        return jsonify({'error': 'No data'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
