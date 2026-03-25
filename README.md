# Cyberbullying Detection & Escalation Prediction 🛡️

A production-ready, full-stack application that detects cyberbullying in messages and predicts conversation escalation risk in real-time. Built with a React dashboard, Node.js WebSocket backend, and Python FastAPI Machine Learning service.

## 🧱 Architecture

1. **Frontend (/frontend):** React, Vite, Tailwind CSS, Chart.js. Provides a live dashboard to monitor monitored conversations, upload training datasets, and view historical analytics.
2. **Backend (/backend):** Node.js, Express, MongoDB, WebSockets. Acts as a middleware, persisting data to MongoDB and broadcasting real-time predictions to the dashboard.
3. **ML Service (/ml-service):** Python, FastAPI, Scikit-learn, NLTK. Handles text preprocessing, logistic regression for detecting bullying, and contextual rule-based/random forest models for predicting escalation.

---

## 🚀 Getting Started (Docker Compose - Recommended)

The easiest way to run the entire stack is using Docker Compose.

1. Install Docker and Docker Compose.
2. Open terminal in the root directory.
3. Run the following command:
   ```bash
   docker-compose up --build
   ```
4. Access the applications:
   - **Frontend UI:** `http://localhost:5173`
   - **Backend API:** `http://localhost:5000`
   - **ML Service Docs:** `http://localhost:8000/docs`



## 💻 Local Setup (Without Docker)

If you prefer to run services manually, follow these steps:

### Prerequisites
- Node.js (v18+)
- Python (3.10+)
- MongoDB server running on `localhost:27017`

### 1. ML Service
```bash
cd ml-service
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Backend
```bash
cd backend
npm install
npm run dev
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🧠 Training the Models

Before the Live Monitor works accurately, you need to train the models:

1. Go to the **Train Data** tab (`http://localhost:5173/upload`).
2. Upload the sample dataset located at `ml-service/data/sample_dataset.csv`.
3. Click **Start Training**.
4. The models will be saved to `ml-service/saved_models/` and are immediately ready for prediction.

---

## 🚨 Features

- **Toxicity Prediction:** Identifies individual abusive or toxic messages using NLP.
- **Contextual Escalation Risk:** Groups messages by conversation to calculate escalation trends (LOW / MEDIUM / HIGH risk).
- **Real-Time WebSockets:** New predictions instantly appear on the dashboard without refreshing.
- **Analytics Dashboard:** Visualizes toxicity trends and risk buckets using Chart.js.
