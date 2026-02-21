"""
MongoDB integration for QuantumTrade PRO
Stores predictions, watchlist, and computes live analytics.
"""

from pymongo import MongoClient, DESCENDING, ASCENDING
from datetime import datetime, timedelta
from bson import ObjectId
import requests
import os

# Load env vars (needed when db.py is imported before app.py's load_dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv('/etc/secrets/.env')  # Render secret files
    load_dotenv()  # Local .env fallback
except ImportError:
    pass

# ============= MongoDB Connection =============
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/quantumtrade_pro")
print(f"[DB] Connecting to MongoDB...")
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

# Get database name from URI or fallback
try:
    db = client.get_default_database()
    print(f"[DB] Using database: {db.name}")
except Exception:
    db = client["quantumtrade_pro"]
    print(f"[DB] Using fallback database: quantumtrade_pro")

# Collections
predictions_col = db["predictions"]
watchlist_col = db["watchlist"]
model_feedback_col = db["model_feedback"]  # Adaptive learning store
users_col = db["users"]  # User accounts

# Create indexes for performance (non-blocking — app starts even if DB is down)
try:
    predictions_col.create_index([("created_at", DESCENDING)])
    predictions_col.create_index([("ticker", ASCENDING)])
    predictions_col.create_index([("signal", ASCENDING)])
    predictions_col.create_index([("outcome", ASCENDING)])
    predictions_col.create_index([("user_id", ASCENDING)])
    watchlist_col.create_index([("ticker", ASCENDING), ("user_id", ASCENDING)], unique=True)
    model_feedback_col.create_index([("ticker", ASCENDING)], unique=True)
    users_col.create_index([("username", ASCENDING)], unique=True)
    users_col.create_index([("email", ASCENDING)], unique=True)
    print("[DB] Indexes created successfully")
except Exception as e:
    print(f"[DB] Warning: Could not create indexes (DB may be unavailable): {e}")


# ============= User Account Functions =============
def create_user(first_name, last_name, username, email, phone, hashed_password, recovery_phrase, position_size=10000):
    """Create a new user account."""
    user = {
        "first_name": first_name,
        "last_name": last_name,
        "username": username,
        "email": email,
        "phone": phone,
        "password": hashed_password,
        "recovery_phrase": recovery_phrase, # 12 numbers for password recovery
        "position_size": position_size,  # User-specified P&L calculation base
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    result = users_col.insert_one(user)
    user["_id"] = result.inserted_id
    return user


def get_user_by_recovery_phrase(phrase_str):
    """Find a user by their unique 12-number recovery phrase."""
    return users_col.find_one({"recovery_phrase": phrase_str})


def get_user_by_username(username):
    """Get user by username."""
    return users_col.find_one({"username": username})


def get_user_by_email(email):
    """Get user by email."""
    return users_col.find_one({"email": email})


def get_user_by_id(user_id):
    """Get user by ObjectId."""
    return users_col.find_one({"_id": ObjectId(user_id)})


def update_user_profile(username, updates):
    """Update user profile fields."""
    updates["updated_at"] = datetime.utcnow()
    return users_col.update_one(
        {"username": username},
        {"$set": updates}
    )


def update_user_password(username, hashed_password):
    """Update user password."""
    return users_col.update_one(
        {"username": username},
        {"$set": {
            "password": hashed_password,
            "password_updated_at": datetime.utcnow()
        }}
    )


def get_user_stats(username):
    """Get prediction statistics for a user."""
    pipeline = [
        {"$match": {"user_id": username}},
        {"$group": {
            "_id": None,
            "total_predictions": {"$sum": 1},
            "correct": {"$sum": {"$cond": [{"$eq": ["$outcome", "correct"]}, 1, 0]}},
            "incorrect": {"$sum": {"$cond": [{"$eq": ["$outcome", "incorrect"]}, 1, 0]}},
            "total_pnl": {"$sum": {"$ifNull": ["$pnl", 0]}},
            "avg_accuracy": {"$avg": "$directional_accuracy"},
        }}
    ]
    result = list(predictions_col.aggregate(pipeline))
    if result:
        r = result[0]
        verified = r["correct"] + r["incorrect"]
        return {
            "total_predictions": r["total_predictions"],
            "correct": r["correct"],
            "incorrect": r["incorrect"],
            "win_rate": round(r["correct"] / verified * 100, 1) if verified > 0 else 0,
            "total_pnl": round(r["total_pnl"], 2),
            "avg_accuracy": round(r["avg_accuracy"] or 0, 1),
        }
    return {
        "total_predictions": 0, "correct": 0, "incorrect": 0,
        "win_rate": 0, "total_pnl": 0, "avg_accuracy": 0,
    }


# ============= Sector Mapping =============
SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'AMZN': 'Technology', 'META': 'Technology', 'NVDA': 'Technology', 'TSLA': 'Technology',
    'NFLX': 'Technology', 'INTC': 'Technology', 'AMD': 'Technology', 'CRM': 'Technology',
    'ADBE': 'Technology', 'ORCL': 'Technology', 'CSCO': 'Technology', 'AVGO': 'Technology',
    'QCOM': 'Technology', 'TXN': 'Technology', 'MU': 'Technology', 'SHOP': 'Technology',
    'SQ': 'Technology', 'SNOW': 'Technology', 'PLTR': 'Technology', 'UBER': 'Technology',
    'ABNB': 'Technology', 'COIN': 'Technology', 'RBLX': 'Technology', 'SNAP': 'Technology',
    'PINS': 'Technology', 'ZM': 'Technology', 'DDOG': 'Technology', 'NET': 'Technology',
    'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance', 'MS': 'Finance',
    'V': 'Finance', 'MA': 'Finance', 'AXP': 'Finance', 'WFC': 'Finance',
    'C': 'Finance', 'BLK': 'Finance', 'SCHW': 'Finance', 'USB': 'Finance',
    'PNC': 'Finance', 'COF': 'Finance', 'BK': 'Finance',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'MRK': 'Healthcare',
    'ABBV': 'Healthcare', 'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
    'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare', 'ISRG': 'Healthcare',
    'MDT': 'Healthcare', 'DHR': 'Healthcare', 'MRNA': 'Healthcare', 'BNTX': 'Healthcare',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'OXY': 'Energy', 'EOG': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy',
    'VLO': 'Energy', 'DVN': 'Energy', 'HAL': 'Energy',
    'WMT': 'Consumer', 'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer',
    'COST': 'Consumer', 'MCD': 'Consumer', 'DIS': 'Consumer', 'NKE': 'Consumer',
    'SBUX': 'Consumer', 'HD': 'Consumer', 'LOW': 'Consumer', 'TGT': 'Consumer',
    'CMG': 'Consumer', 'YUM': 'Consumer', 'EL': 'Consumer',
    'SPY': 'Index/ETF', 'QQQ': 'Index/ETF', 'DIA': 'Index/ETF', 'IWM': 'Index/ETF',
    'VOO': 'Index/ETF', 'VTI': 'Index/ETF', 'ARKK': 'Index/ETF',
}


def get_sector(ticker):
    """Get sector for a ticker, default to 'Other'"""
    return SECTOR_MAP.get(ticker.upper(), 'Other')


# ============= Prediction Storage =============
def save_prediction(prediction_data):
    """
    Save a prediction to MongoDB.
    prediction_data should contain all the metrics from the predict route.
    """
    doc = {
        "ticker": prediction_data["ticker"],
        "signal": prediction_data["signal"],
        "signal_class": prediction_data["signal_class"],
        "confidence": prediction_data["confidence"],
        "directional_accuracy": prediction_data["directional_accuracy"],
        "mape": prediction_data["mape"],
        "r2": prediction_data["r2"],
        "mse": prediction_data["mse"],
        "mae": prediction_data["mae"],
        "rmse": prediction_data["rmse"],
        "lstm_accuracy": prediction_data["lstm_accuracy"],
        "xgb_accuracy": prediction_data["xgb_accuracy"],
        "lstm_weight": prediction_data["lstm_weight"],
        "xgb_weight": prediction_data["xgb_weight"],
        "num_features": prediction_data["num_features"],
        "data_points": prediction_data["data_points"],
        "current_price": prediction_data["current_price"],
        "predicted_prices": prediction_data["predicted_prices"],  # list of 30 future prices
        "predicted_direction": prediction_data["predicted_direction"],  # "up" or "down"
        "sector": get_sector(prediction_data["ticker"]),
        "user_id": prediction_data.get("user_id"),  # linked to user account
        "position_size": prediction_data.get("position_size", 10000),  # user's position size
        "created_at": datetime.utcnow(),
        "outcome": None,        # "correct", "incorrect", or None (pending)
        "outcome_checked_at": None,
        "actual_price_after": None,
        "pnl": None,            # calculated P/L
    }
    result = predictions_col.insert_one(doc)
    return str(result.inserted_id)



# ============= Outcome Verification =============
def check_prediction_outcomes(polygon_api_key):
    """
    Check predictions older than 1 trading day and update their outcomes
    by comparing predicted direction with actual price movement.
    """
    cutoff = datetime.utcnow() - timedelta(days=1)
    pending = predictions_col.find({
        "outcome": None,
        "created_at": {"$lt": cutoff}
    }).limit(50)  # batch limit

    for pred in pending:
        try:
            # Get current price for the ticker
            ticker = pred["ticker"]
            end = datetime.today().strftime('%Y-%m-%d')
            start = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
                f"{start}/{end}?adjusted=true&sort=desc&limit=1&apiKey={polygon_api_key}"
            )
            resp = requests.get(url, timeout=10).json()
            if 'results' not in resp or len(resp['results']) == 0:
                continue

            actual_price = resp['results'][0]['c']
            original_price = pred["current_price"]
            predicted_dir = pred["predicted_direction"]

            # Determine if prediction was correct
            actual_dir = "up" if actual_price > original_price else "down"
            is_correct = (predicted_dir == actual_dir)

            # Calculate P/L using user's position size (stored in prediction) or default
            position_size = pred.get("position_size", 10000)
            price_change_pct = (actual_price - original_price) / original_price


            if pred["signal"] in ["STRONG BUY", "BUY"]:
                pnl = round(position_size * price_change_pct, 2)
            elif pred["signal"] in ["STRONG SELL", "SELL"]:
                pnl = round(position_size * (-price_change_pct), 2)
            else:  # HOLD
                pnl = 0

            predictions_col.update_one(
                {"_id": pred["_id"]},
                {"$set": {
                    "outcome": "correct" if is_correct else "incorrect",
                    "outcome_checked_at": datetime.utcnow(),
                    "actual_price_after": actual_price,
                    "pnl": pnl,
                }}
            )

            # ===== ADAPTIVE LEARNING: Feed outcome back =====
            update_adaptive_feedback(
                ticker=ticker,
                signal=pred["signal"],
                is_correct=is_correct,
                lstm_accuracy=pred.get("lstm_accuracy", 0),
                xgb_accuracy=pred.get("xgb_accuracy", 0),
                lstm_weight=pred.get("lstm_weight", 50),
                xgb_weight=pred.get("xgb_weight", 50),
            )

        except Exception as e:
            print(f"Error checking outcome for {pred.get('ticker', '?')}: {e}")
            continue


# ============= Adaptive Learning System =============
def get_adaptive_params(ticker):
    """
    Get adaptive learning parameters for a specific ticker.
    Returns adjusted weights, confidence modifiers, and signal reliability
    based on the model's historical performance on this ticker.
    """
    ticker = ticker.upper().strip()
    feedback = model_feedback_col.find_one({"ticker": ticker})

    # Default params (no learning data yet)
    defaults = {
        "has_history": False,
        "ticker": ticker,
        "lstm_weight_adj": 0.5,     # 50/50 default
        "xgb_weight_adj": 0.5,
        "confidence_modifier": 1.0,  # no adjustment
        "signal_reliability": {
            "BUY": 1.0, "STRONG BUY": 1.0,
            "SELL": 1.0, "STRONG SELL": 1.0,
            "HOLD": 1.0,
        },
        "total_outcomes": 0,
        "overall_win_rate": 0,
        "avg_lstm_accuracy": 50,
        "avg_xgb_accuracy": 50,
        "best_model": "ensemble",
        "learning_note": "No historical data yet — using default parameters.",
    }

    if not feedback or feedback.get("total_outcomes", 0) < 2:
        return defaults

    # Calculate adaptive weights from historical performance
    avg_lstm = feedback.get("avg_lstm_accuracy", 50)
    avg_xgb = feedback.get("avg_xgb_accuracy", 50)
    total_acc = avg_lstm + avg_xgb + 1e-10
    lstm_w = avg_lstm / total_acc
    xgb_w = avg_xgb / total_acc

    # Determine best model
    if avg_lstm > avg_xgb + 5:
        best_model = "LSTM"
    elif avg_xgb > avg_lstm + 5:
        best_model = "XGBoost"
    else:
        best_model = "Ensemble"

    # Signal reliability: win rate per signal type
    signal_rel = {}
    signal_data = feedback.get("signal_stats", {})
    for sig, stats in signal_data.items():
        total = stats.get("total", 0)
        correct = stats.get("correct", 0)
        if total >= 2:
            win_rate = correct / total
            # Scale: 0.5 = unreliable, 1.0 = perfectly reliable
            signal_rel[sig] = round(max(0.3, min(1.5, win_rate * 2)), 2)
        else:
            signal_rel[sig] = 1.0  # not enough data

    # Fill missing signals
    for sig in ["BUY", "STRONG BUY", "SELL", "STRONG SELL", "HOLD"]:
        if sig not in signal_rel:
            signal_rel[sig] = 1.0

    # Overall confidence modifier based on total correctness
    overall_win = feedback.get("correct_count", 0)
    overall_total = feedback.get("total_outcomes", 1)
    overall_win_rate = overall_win / overall_total
    # Scale: <40% → lower confidence, >60% → higher confidence
    confidence_mod = round(max(0.5, min(1.5, overall_win_rate * 2)), 2)

    # Build learning note
    notes = []
    if overall_win_rate >= 0.6:
        notes.append(f"Model has {overall_win_rate*100:.0f}% win rate on {ticker} — high confidence.")
    elif overall_win_rate >= 0.4:
        notes.append(f"Model has {overall_win_rate*100:.0f}% win rate on {ticker} — moderate confidence.")
    else:
        notes.append(f"Model has only {overall_win_rate*100:.0f}% win rate on {ticker} — predictions unreliable.")

    if best_model != "Ensemble":
        notes.append(f"{best_model} performs better for {ticker} — weighting adjusted.")

    # Check if any signal type is unreliable
    for sig, rel in signal_rel.items():
        if rel < 0.7 and sig in signal_data and signal_data[sig].get("total", 0) >= 3:
            notes.append(f"⚠️ '{sig}' signals have been unreliable for {ticker}.")

    return {
        "has_history": True,
        "ticker": ticker,
        "lstm_weight_adj": round(lstm_w, 3),
        "xgb_weight_adj": round(xgb_w, 3),
        "confidence_modifier": confidence_mod,
        "signal_reliability": signal_rel,
        "total_outcomes": overall_total,
        "overall_win_rate": round(overall_win_rate * 100, 1),
        "avg_lstm_accuracy": round(avg_lstm, 1),
        "avg_xgb_accuracy": round(avg_xgb, 1),
        "best_model": best_model,
        "learning_note": " ".join(notes),
    }


def update_adaptive_feedback(ticker, signal, is_correct, lstm_accuracy, xgb_accuracy, lstm_weight, xgb_weight):
    """
    Update the adaptive learning store after an outcome is verified.
    This is the feedback loop that makes the model smarter over time.
    """
    ticker = ticker.upper().strip()

    # Get existing feedback or create new
    existing = model_feedback_col.find_one({"ticker": ticker})

    if existing:
        # Running averages (exponential moving average for recency bias)
        alpha = 0.3  # Weight recent outcomes more heavily
        old_lstm = existing.get("avg_lstm_accuracy", 50)
        old_xgb = existing.get("avg_xgb_accuracy", 50)
        new_lstm_avg = round(alpha * lstm_accuracy + (1 - alpha) * old_lstm, 2)
        new_xgb_avg = round(alpha * xgb_accuracy + (1 - alpha) * old_xgb, 2)

        total_outcomes = existing.get("total_outcomes", 0) + 1
        correct_count = existing.get("correct_count", 0) + (1 if is_correct else 0)

        # Update per-signal stats
        signal_stats = existing.get("signal_stats", {})
        if signal not in signal_stats:
            signal_stats[signal] = {"total": 0, "correct": 0}
        signal_stats[signal]["total"] += 1
        if is_correct:
            signal_stats[signal]["correct"] += 1

        model_feedback_col.update_one(
            {"ticker": ticker},
            {"$set": {
                "avg_lstm_accuracy": new_lstm_avg,
                "avg_xgb_accuracy": new_xgb_avg,
                "total_outcomes": total_outcomes,
                "correct_count": correct_count,
                "signal_stats": signal_stats,
                "last_updated": datetime.utcnow(),
            }}
        )
    else:
        # First outcome for this ticker
        signal_stats = {
            signal: {"total": 1, "correct": 1 if is_correct else 0}
        }
        model_feedback_col.insert_one({
            "ticker": ticker,
            "avg_lstm_accuracy": float(lstm_accuracy),
            "avg_xgb_accuracy": float(xgb_accuracy),
            "total_outcomes": 1,
            "correct_count": 1 if is_correct else 0,
            "signal_stats": signal_stats,
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
        })

    print(f"  [Adaptive] {ticker}: Updated feedback — {'✓ correct' if is_correct else '✗ incorrect'} | "
          f"Total outcomes: {existing.get('total_outcomes', 0) + 1 if existing else 1}")


# ============= Analytics Queries =============
def get_analytics_data(user_id=None):
    """Get all analytics data for the analytics page."""
    filter_q = {"user_id": user_id} if user_id else {}
    total = predictions_col.count_documents(filter_q)

    if total == 0:
        return {
            "total_predictions": 0,
            "avg_accuracy": 0,
            "avg_mape": 0,
            "total_pnl": 0,
            "watchlist_count": watchlist_col.count_documents(filter_q),
            "accuracy_trend": [],
            "signal_distribution": {"STRONG BUY": 0, "BUY": 0, "HOLD": 0, "SELL": 0, "STRONG SELL": 0},
            "sector_performance": {},
            "model_performance": {"lstm_avg": 0, "xgb_avg": 0, "ensemble_avg": 0},
            "recent_predictions": [],
            "correct_count": 0,
            "incorrect_count": 0,
            "pending_count": 0,
            "pnl_change_pct": 0,
            "accuracy_change": 0,
        }

    # Aggregate average accuracy
    pipeline_avg = [
        {"$match": filter_q},
        {"$group": {
            "_id": None,
            "avg_accuracy": {"$avg": "$directional_accuracy"},
            "avg_mape": {"$avg": "$mape"},
            "avg_lstm": {"$avg": "$lstm_accuracy"},
            "avg_xgb": {"$avg": "$xgb_accuracy"},
            "total_pnl": {"$sum": {"$ifNull": ["$pnl", 0]}},
        }}
    ]
    avg_result = list(predictions_col.aggregate(pipeline_avg))
    avg_data = avg_result[0] if avg_result else {}

    # Outcome counts
    correct = predictions_col.count_documents({**filter_q, "outcome": "correct"})
    incorrect = predictions_col.count_documents({**filter_q, "outcome": "incorrect"})
    pending = predictions_col.count_documents({**filter_q, "outcome": None})

    # Signal distribution
    signal_pipeline = [
        {"$match": filter_q},
        {"$group": {"_id": "$signal", "count": {"$sum": 1}}}
    ]
    signal_dist = {s["_id"]: s["count"] for s in predictions_col.aggregate(signal_pipeline)}

    # Accuracy trend (daily average, last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    trend_pipeline = [
        {"$match": {**filter_q, "created_at": {"$gte": thirty_days_ago}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
            "avg_acc": {"$avg": "$directional_accuracy"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}}
    ]
    accuracy_trend = list(predictions_col.aggregate(trend_pipeline))

    # Sector performance (win rate by sector)
    sector_pipeline = [
        {"$match": {**filter_q, "outcome": {"$ne": None}}},
        {"$group": {
            "_id": "$sector",
            "total": {"$sum": 1},
            "wins": {"$sum": {"$cond": [{"$eq": ["$outcome", "correct"]}, 1, 0]}},
            "avg_accuracy": {"$avg": "$directional_accuracy"},
        }}
    ]
    sector_stats = list(predictions_col.aggregate(sector_pipeline))
    sector_performance = {}
    for s in sector_stats:
        if s["_id"] and s["total"] > 0:
            sector_performance[s["_id"]] = {
                "win_rate": round((s["wins"] / s["total"]) * 100),
                "total": s["total"],
                "avg_accuracy": round(s["avg_accuracy"], 1)
            }

    # Recent predictions (last 5)
    recent = list(predictions_col.find(filter_q).sort("created_at", DESCENDING).limit(5))
    for r in recent:
        r["_id"] = str(r["_id"])
        r["created_at"] = r["created_at"].strftime("%b %d, %Y %H:%M")

    # Calculate changes (compare last 7 days with previous 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    fourteen_days_ago = datetime.utcnow() - timedelta(days=14)

    recent_preds = predictions_col.count_documents({**filter_q, "created_at": {"$gte": seven_days_ago}})
    older_preds = predictions_col.count_documents({
        **filter_q,
        "created_at": {"$gte": fourteen_days_ago, "$lt": seven_days_ago}
    })
    pred_change = recent_preds - older_preds

    # Average accuracy for this week vs last week
    this_week_acc = list(predictions_col.aggregate([
        {"$match": {**filter_q, "created_at": {"$gte": seven_days_ago}}},
        {"$group": {"_id": None, "avg": {"$avg": "$directional_accuracy"}}}
    ]))
    last_week_acc = list(predictions_col.aggregate([
        {"$match": {**filter_q, "created_at": {"$gte": fourteen_days_ago, "$lt": seven_days_ago}}},
        {"$group": {"_id": None, "avg": {"$avg": "$directional_accuracy"}}}
    ]))
    tw_avg = this_week_acc[0]["avg"] if this_week_acc else 0
    lw_avg = last_week_acc[0]["avg"] if last_week_acc else 0
    accuracy_change = round(tw_avg - lw_avg, 1) if lw_avg else 0

    ensemble_avg = avg_data.get("avg_accuracy", 0)
    lstm_avg = avg_data.get("avg_lstm", 0)
    xgb_avg = avg_data.get("avg_xgb", 0)

    return {
        "total_predictions": total,
        "avg_accuracy": round(avg_data.get("avg_accuracy", 0), 1),
        "avg_mape": round(avg_data.get("avg_mape", 0), 2),
        "total_pnl": round(avg_data.get("total_pnl", 0), 2),
        "watchlist_count": watchlist_col.count_documents(filter_q),
        "accuracy_trend": accuracy_trend,
        "signal_distribution": {
            "STRONG BUY": signal_dist.get("STRONG BUY", 0),
            "BUY": signal_dist.get("BUY", 0),
            "HOLD": signal_dist.get("HOLD", 0),
            "SELL": signal_dist.get("SELL", 0),
            "STRONG SELL": signal_dist.get("STRONG SELL", 0),
        },
        "sector_performance": sector_performance,
        "model_performance": {
            "lstm_avg": round(lstm_avg, 1) if lstm_avg else 0,
            "xgb_avg": round(xgb_avg, 1) if xgb_avg else 0,
            "ensemble_avg": round(ensemble_avg, 1) if ensemble_avg else 0,
        },
        "recent_predictions": recent,
        "correct_count": correct,
        "incorrect_count": incorrect,
        "pending_count": pending,
        "pnl_change_pct": 5.2, # dummy
        "accuracy_change": accuracy_change,
        "pred_change": pred_change,
    }

# ============= History Queries =============

def get_history_data(user_id=None, page=1, per_page=15, signal_filter=None, min_accuracy=None, outcome_filter=None):
    """Get paginated prediction history with filters filtering by user_id."""
    query = {"user_id": user_id} if user_id else {}

    if signal_filter and signal_filter != "all":
        query["signal"] = signal_filter

    if min_accuracy:
        try:
            query["directional_accuracy"] = {"$gte": float(min_accuracy)}
        except (ValueError, TypeError):
            pass

    if outcome_filter and outcome_filter != "all":
        if outcome_filter == "pending":
            query["outcome"] = None
        else:
            query["outcome"] = outcome_filter

    total = predictions_col.count_documents(query)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, total_pages)
    skip = (page - 1) * per_page

    results = list(
        predictions_col.find(query)
        .sort("created_at", DESCENDING)
        .skip(skip)
        .limit(per_page)
    )

    for r in results:
        r["_id"] = str(r["_id"])
        r["created_at_fmt"] = r["created_at"].strftime("%b %d, %Y")

    # Summary stats (across ALL matching predictions, not just current page)
    summary_pipeline = [
        {"$match": query if query else {}},
        {"$group": {
            "_id": None,
            "total": {"$sum": 1},
            "avg_accuracy": {"$avg": "$directional_accuracy"},
            "avg_mape": {"$avg": "$mape"},
            "total_pnl": {"$sum": {"$ifNull": ["$pnl", 0]}},
            "correct": {"$sum": {"$cond": [{"$eq": ["$outcome", "correct"]}, 1, 0]}},
            "incorrect": {"$sum": {"$cond": [{"$eq": ["$outcome", "incorrect"]}, 1, 0]}},
        }}
    ]
    summary = list(predictions_col.aggregate(summary_pipeline))
    summary_data = summary[0] if summary else {
        "total": 0, "avg_accuracy": 0, "avg_mape": 0, "total_pnl": 0,
        "correct": 0, "incorrect": 0
    }

    return {
        "predictions": results,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "summary": {
            "total": summary_data.get("total", 0),
            "avg_accuracy": round(summary_data.get("avg_accuracy", 0), 1),
            "avg_mape": round(summary_data.get("avg_mape", 0), 2),
            "total_pnl": round(summary_data.get("total_pnl", 0), 2),
            "correct": summary_data.get("correct", 0),
            "incorrect": summary_data.get("incorrect", 0),
        }
    }


# ============= Watchlist (Portfolio) =============
def add_to_watchlist(user_id, ticker, entry_price, polygon_api_key):
    """Add a stock to the watchlist for a specific user."""
    ticker = ticker.upper().strip()
    if not ticker or not user_id:
        return None

    # If entry_price not provided, fetch current
    if not entry_price or entry_price <= 0:
        entry_price = _fetch_current_price(ticker, polygon_api_key)
        if entry_price is None:
            return None

    doc = {
        "ticker": ticker,
        "user_id": user_id,
        "entry_price": round(entry_price, 2),
        "sector": get_sector(ticker),
        "added_at": datetime.utcnow(),
    }

    try:
        watchlist_col.update_one(
            {"ticker": ticker, "user_id": user_id},
            {"$set": doc},
            upsert=True
        )
        return doc
    except Exception as e:
        print(f"Error adding to watchlist: {e}")
        return None


def remove_from_watchlist(user_id, ticker):
    """Remove a stock from the user's watchlist."""
    result = watchlist_col.delete_one({
        "ticker": ticker.upper().strip(),
        "user_id": user_id
    })
    return result.deleted_count > 0


def get_watchlist(user_id=None):
    """Get all watchlist items for a specific user."""
    filter_q = {"user_id": user_id} if user_id else {}
    return list(watchlist_col.find(filter_q).sort("added_at", DESCENDING))


def get_portfolio_data(user_id, polygon_api_key):
    """Get full portfolio data for a user with live prices and computed P/L."""
    watchlist = get_watchlist(user_id)

    if not watchlist:
        return {
            "stocks": [],
            "total_value": 0,
            "total_invested": 0,
            "total_pnl": 0,
            "total_pnl_pct": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "stock_count": 0,
            "alerts": [],
        }

    stocks = []
    total_invested = 0
    total_current = 0
    win_count = 0
    loss_count = 0

    for item in watchlist:
        ticker = item["ticker"]
        entry_price = item["entry_price"]
        current_price = _fetch_current_price(ticker, polygon_api_key)

        if current_price is None:
            current_price = entry_price  # fallback

        pnl = round(current_price - entry_price, 2)
        pnl_pct = round(((current_price - entry_price) / entry_price) * 100, 1) if entry_price > 0 else 0

        # Get latest prediction for this stock
        latest_pred = predictions_col.find_one(
            {"ticker": ticker},
            sort=[("created_at", DESCENDING)]
        )

        signal = latest_pred.get("signal", "—") if latest_pred else "—"
        signal_class = latest_pred.get("signal_class", "hold") if latest_pred else "hold"
        accuracy = latest_pred.get("directional_accuracy", 0) if latest_pred else 0

        # Position sizing: 1 share for simplicity
        invested = entry_price
        current_val = current_price
        total_invested += invested
        total_current += current_val

        if pnl > 0:
            win_count += 1
        elif pnl < 0:
            loss_count += 1

        stocks.append({
            "ticker": ticker,
            "entry_price": entry_price,
            "current_price": round(current_price, 2),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "signal": signal,
            "signal_class": signal_class,
            "accuracy": round(accuracy, 1) if accuracy else 0,
            "sector": item.get("sector", "Other"),
            "added_at": item["added_at"].strftime("%b %d, %Y"),
        })

    total_pnl = round(total_current - total_invested, 2)
    total_pnl_pct = round(((total_current - total_invested) / total_invested) * 100, 1) if total_invested > 0 else 0
    total_trades = win_count + loss_count
    win_rate = round((win_count / total_trades) * 100, 1) if total_trades > 0 else 0
    avg_pnl = round(total_pnl / len(stocks), 2) if stocks else 0

    # Generate alerts from recent predictions on watchlist stocks
    watchlist_tickers = [s["ticker"] for s in stocks]
    recent_preds = list(predictions_col.find(
        {"ticker": {"$in": watchlist_tickers}}
    ).sort("created_at", DESCENDING).limit(5))

    alerts = []
    for pred in recent_preds:
        signal = pred.get("signal", "HOLD")
        ticker = pred["ticker"]
        accuracy = pred.get("directional_accuracy", 0)
        age = datetime.utcnow() - pred["created_at"]

        if age.total_seconds() < 3600:
            time_str = f"{int(age.total_seconds() / 60)}m ago"
        elif age.total_seconds() < 86400:
            time_str = f"{int(age.total_seconds() / 3600)}h ago"
        else:
            time_str = f"{age.days}d ago"

        if "BUY" in signal:
            icon = "▲"
            color = "green"
            desc = f"LSTM + XGBoost ensemble shows upward momentum with {accuracy:.0f}% accuracy"
        elif "SELL" in signal:
            icon = "▼"
            color = "red"
            desc = f"Bearish signals detected — model suggests downward pressure with {accuracy:.0f}% accuracy"
        else:
            icon = "◆"
            color = "gold"
            desc = f"Consolidation pattern — model predicts sideways movement with {accuracy:.0f}% accuracy"

        alerts.append({
            "ticker": ticker,
            "signal": signal,
            "icon": icon,
            "color": color,
            "description": desc,
            "time": time_str,
        })

    return {
        "stocks": stocks,
        "total_value": round(total_current, 2),
        "total_invested": round(total_invested, 2),
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "stock_count": len(stocks),
        "alerts": alerts,
    }


# ============= Homepage Stats =============
def get_homepage_stats():
    """Get dynamic stats for the homepage."""
    total = predictions_col.count_documents({})
    if total == 0:
        return {
            "avg_accuracy": 80,
            "total_features": 35,
            "data_years": 10,
            "model_count": 2,
        }

    pipeline = [
        {"$group": {
            "_id": None,
            "avg_accuracy": {"$avg": "$directional_accuracy"},
            "avg_features": {"$avg": "$num_features"},
        }}
    ]
    result = list(predictions_col.aggregate(pipeline))
    data = result[0] if result else {}

    return {
        "avg_accuracy": round(data.get("avg_accuracy", 80), 1),
        "total_features": round(data.get("avg_features", 35)),
        "data_years": 3,
        "model_count": 2,
    }


# ============= Helper =============
def _fetch_current_price(ticker, polygon_api_key):
    """Fetch the latest closing price for a ticker from Polygon.io"""
    try:
        end = datetime.today().strftime('%Y-%m-%d')
        start = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start}/{end}?adjusted=true&sort=desc&limit=1&apiKey={polygon_api_key}"
        )
        resp = requests.get(url, timeout=10).json()
        if 'results' in resp and len(resp['results']) > 0:
            return resp['results'][0]['c']
        return None
    except Exception:
        return None
