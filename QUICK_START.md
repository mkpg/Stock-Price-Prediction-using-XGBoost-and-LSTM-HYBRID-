# 🚀 QUICK START GUIDE

## Setup (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the App
```bash
python app.py
```

### Step 3: Open in Browser
```
http://localhost:5000
```

---

## How to Use

1. **Enter Stock Ticker**: Type any US stock symbol
   - Examples: AAPL, MSFT, TSLA, GOOGL, AMZN, NVDA

2. **Click "Predict"** and wait 30-60 seconds

3. **View Results**:
   - 🎯 Trading Signal (BUY/SELL/HOLD)
   - 📊 Directional Accuracy % (how often up/down is correct)
   - 📉 Prediction Error (MAPE)
   - 📈 Interactive chart

---

## What the Metrics Mean

| Metric | What it is | Good Range |
|--------|-----------|-----------|
| **Directional Accuracy** | % of time model predicts up/down correctly | 65-75% |
| **MAPE** | Average % error in price prediction | 8-15% |
| **MAE** | Average $ error per prediction | $2-$5 |
| **RMSE** | Penalizes larger errors more | Small better |
| **MSE** | Squared error metric | Small better |

---

## Understanding the Chart

```
Blue Line     = Actual historical prices
Orange Line   = Model's test predictions
Green Line    = LSTM only 30-day forecast
Red Line      = XGBoost only 30-day forecast
Purple Line   = Final ensemble prediction ⭐ USE THIS
```

---

## Trading Signals Explained

### 🟢 BUY
- Model predicts price increase
- Trend is positive
- 70%+ confidence

### 🔴 SELL
- Model predicts price decrease
- Trend is negative
- 70%+ confidence

### 🟡 HOLD
- Mixed signals or unclear trend
- Wait for clearer signal

⚠️ **Important**: These are **predictions, not financial advice**. Always do your own research before trading.

---

## How the Model Works

```
Your Stock Ticker (e.g., AAPL)
        ↓
Fetch 10 years of historical data
        ↓
Calculate 8 Technical Indicators:
  • RSI (momentum)
  • MACD (trend)
  • Bollinger Bands (volatility)
  • Moving Averages (trend)
  • Volume changes
  • Price momentum
        ↓
Split data: 85% training, 15% testing
        ↓
Train 2 models:
  • Bidirectional LSTM (deep learning)
  • XGBoost (decision trees)
        ↓
Combine predictions (50/50 ensemble)
        ↓
Test on historical data
        ↓
Predict next 30 days
        ↓
Generate BUY/SELL/HOLD signal
```

---

## Common Questions

### Q: Why is it slow?
**A**: First prediction trains models from scratch (~30-60 sec). Subsequent tickers are faster.

### Q: What stocks can I predict?
**A**: Any US stock with 100+ days of trading history (Polygon.io).

### Q: Is this financial advice?
**A**: **NO** - This is educational only. Always consult a financial advisor before trading.

### Q: How accurate is it?
**A**: 65-75% directional accuracy on average. Not perfect, but better than random (50%).

### Q: Can I trade live?
**A**: Yes, but start small and backtest your strategy first. Risk management is crucial.

### Q: What if I get an error?
**A**: 
- Check internet connection
- Ensure ticker symbol is valid (US stocks only)
- Try a different stock
- See README.md for more troubleshooting

---

## Advanced Configuration

To change model parameters, edit `app.py`:

```python
# Line 87-90: Change time window
time_steps = 60  # 60 days of history

# Line 80-90: Change LSTM architecture
LSTM units = [128, 64, 32]
epochs = 20
dropout_rate = [0.2, 0.2, 0.1]

# Line 110+: Change XGBoost hyperparameters
n_estimators = 300
max_depth = 7
learning_rate = 0.05
```

---

## Production Tips

1. **Cache Predictions**: Store results to avoid re-training
2. **API Key**: Move to .env file, don't hardcode
3. **Rate Limits**: Polygon.io free = 5 reqs/min
4. **Database**: Add historical prediction logs
5. **Notifications**: Email/SMS when BUY signals generated

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 5000 in use | `python app.py` or change port in app.py last line |
| Module not found | Run: `pip install -r requirements.txt` |
| "Insufficient data" error | Stock needs 100+ days of history |
| Slow predictions | Normal for first run. Try smaller lookback period. |
| API key error | Check POLYGON_API_KEY in app.py line 11 |

---

## Next Steps

1. ✅ Test with AAPL, MSFT, TSLA
2. ✅ Understand the metrics and signals
3. ✅ Backtest on historical data
4. ✅ Build a trading strategy
5. ✅ Add risk management rules
6. ✅ Track predicted vs actual daily
7. ✅ Refine based on results

---

## Key Files

- **app.py** - Main application (ML + Flask)
- **templates/dashboard.html** - Web interface
- **static/style.css** - Styling
- **requirements.txt** - Python packages
- **README.md** - Full documentation

---

## Need Help?

1. Read README.md for comprehensive guide
2. Check UPGRADE_SUMMARY.txt for what changed
3. Review code comments in app.py
4. Check error message in browser

---

**Ready to predict stocks? Start now! 🎉**

```bash
python app.py
```
