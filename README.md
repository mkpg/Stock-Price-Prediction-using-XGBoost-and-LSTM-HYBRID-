---
title: QuantumTrade PRO
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# QuantumTrade PRO - AI Stock Prediction
Premium stock analysis engine using LSTM and XGBoost ensemble models.

### Deployment on Hugging Face Spaces:
This app runs in a Docker container on Hugging Face with 16GB RAM.

### Setup:
Ensure you add the following **Secrets** in the Space Settings:
- `POLYGON_API_KEY`: Your API key from polygon.io
- `MONGO_URI`: Your MongoDB Atlas connection string
- `SECRET_KEY`: A secure random string for Flask sessions
