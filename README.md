# 🛡️ FinSecure AI — Real-Time Fraud Detection System

> An end-to-end, production-ready machine learning pipeline for detecting fraudulent financial transactions in real time with high precision, high recall, and automated AI-driven risk explanations.

📊 **Dataset:** [PaySim Synthetic Financial Datasets for Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/ealaxi/paysim1)

---

## 📌 Project Overview

**FinSecure AI** analyzes complex financial transaction patterns to catch fraudulent activity instantly (<100ms latency). Engineered on a highly imbalanced dataset of 6.3 million records, the system pairs an optimized **XGBoost Classifier** with **OpenAI GPT-3.5** to provide actionable natural language explanations for fraud analysts alongside each classification score.

### Key Features

- ⚡ **Ultra-Low Latency:** Sub-100ms real-time scoring.
- 🎯 **High Performance:** 99.98% AUC-ROC, 80.34% Precision at 99.20% Recall on severe class imbalance (0.13% fraud rate).
- 🧠 **AI-Powered Explanations:** Automated, analyst-friendly risk breakdowns via OpenAI GPT-3.5.
- 🚀 **Production-Ready API:** Asynchronous web services built with FastAPI & Uvicorn.
- 💻 **Interactive UI:** Web dashboard for manual inspection and testing.

---

## 📈 Performance Metrics

| **Metric** | **Score** | **Detail** |
|------------|-----------|------------|
| **AUC-ROC** | **99.98%** | Excellent separation power across thresholds |
| **Precision** | **80.34%** | Low false-positive rate for operational efficiency |
| **Recall** | **99.20%** | Catches over 99% of actual fraudulent transactions |
| **Dataset Size** | **6.3M** | Transactions evaluated via time-based split (70/15/15) |

---

## 🛠️ Tech Stack

- **ML Framework:** XGBoost
- **Data Engineering:** Pandas, NumPy, Scikit-Learn
- **Backend API:** FastAPI, Uvicorn
- **AI Explanations:** OpenAI API (GPT-3.5)
- **Environment:** Python 3.9+

---

## 🚀 Quick Start

### 1. Clone & Navigate

```bash
git clone https://github.com/your-username/finsecure-ai.git
cd finsecure-ai
```
### 2.Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Environment Configuration
Create a .env file in the root directory:
```bash
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```
**Note:** Never commit your .env file or expose your OpenAI API key publicly.

### 4. Run Server
```bash
uvicorn api:app --reload
```

### 5. Access Application

- **Web Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger OpenAPI Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## ⚙️ How It Works

### 1. Data Pipeline & Split

- Engineered on **6.3 Million** transaction logs.
- Enforced **Time-based Splitting** (70% Train / 15% Validation / 15% Test) to prevent temporal data leakage.

### 2. Feature Engineering (15 Key Behavioral Features)

- **Account Draining Metrics:** `oldbalanceOrg`, `newbalanceOrig`, `delta_balance_orig`
- **Transaction Velocity & Recency:** `amount`, `log_amount`, `recency_hours`, `txn_count_24h`
- **Account History Flags:** `is_dest_new`, `is_zero_balance_dest`
- **One-Hot Encoded Types:** One-hot representations for `TRANSFER`, `CASH_OUT`, `PAYMENT`, `DEBIT`, `CASH_IN`

### 3. Model & Threshold Optimization

- **Algorithm:** Tuned XGBoost Classifier with custom class weighting (`scale_pos_weight`).
- **Decision Threshold:** Shifted from standard `0.5` to `0.8` to optimize precision while maintaining high recall.

---

## 📡 API Reference

### Predict Transaction Fraud

`POST /predict`

#### Request Payload Example

```json
{
  "type": "TRANSFER",
  "amount": 180000.00,
  "oldbalanceOrg": 180000.00,
  "newbalanceOrig": 0.00,
  "oldbalanceDest": 0.00,
  "newbalanceDest": 0.00,
  "is_dest_new": 1,
  "txn_count_24h": 4
}
```

#### Response Payload Example

```json
{
  "is_fraud": true,
  "fraud_probability": 0.9642,
  "risk_level": "HIGH",
  "explanation": "High risk detected: The origin account was completely drained to 0 in a single TRANSFER to a new destination recipient with zero prior history."
}
```
## 🔍 Key Domain Insights

### 1. Account Draining Pattern

Fraudulent activity heavily correlates with instant zeroing of the origin account balance.

### 2. High-Risk Transaction Types

Over 95% of fraud occurs exclusively within `TRANSFER` and `CASH_OUT` channels.

### 3. New Recipient Anomaly

High-value transfers directed to unrecognized destination accounts serve as strong positive signals.

### 4. Velocity Bursting

Multiple rapid actions within short temporal windows (`txn_count_24h`) strongly signal automated attack vectors.

---

## 🔮 Future Enhancements

- Integrated model monitoring & real-time feature drift detection using Evidently AI.
- Automated A/B testing framework for model deployment updates.
- Native SHAP (SHapley Additive exPlanations) integration for local feature attribution.
- Kafka integration for event-driven real-time streaming inference.
- Continuous active learning loop with analyst feedback loops.

---

## 📂 Project Structure

```text
finsecure-ai/
│
├── api.py
├── requirements.txt
├── .gitignore
│
├── models/
│   └── fraud_model.pkl
│
├── data/
│   └── processed/
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── templates/
│   └── index.html
│
└── README.md

```

**Note:** Update the project structure above according to the actual files and folders in your repository.

## 🔐 Security

- API keys are stored using environment variables.
- Sensitive credentials are excluded from version control using `.gitignore`.
- Transaction data is processed through a machine learning inference pipeline.
- Fraud predictions are returned with both probability scores and risk classifications.

### Recommended `.gitignore`

```gitignore
.env
__pycache__/
*.pyc
.venv/
venv/
env/
.ipynb_checkpoints/

```

## 🚀 Deployment

FinSecure AI can be deployed using cloud platforms that support Python and FastAPI applications.

Example production server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
For production deployment, environment variables should be configured through the hosting platform rather than committing secrets to the repository.

## 📊 Machine Learning Pipeline

```text
Raw Transaction Data
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Time-Based Train / Validation / Test Split
        │
        ▼
XGBoost Classifier
        │
        ▼
Probability Prediction
        │
        ▼
Optimized Decision Threshold
        │
        ├───────────────┐
        ▼               ▼
 Fraud Detection    Risk Score
        │               │
        └───────┬───────┘
                ▼
       AI Risk Explanation
                │
                ▼
         FastAPI Response
                │
                ▼
        Interactive Dashboard
```

## 📌 Example Prediction Flow

```text
Transaction
     │
     ▼
Feature Extraction
     │
     ▼
XGBoost Model
     │
     ▼
Fraud Probability
     │
     ├── < Threshold ──► LOW / NORMAL
     │
     └── ≥ Threshold ──► HIGH RISK
                              │
                              ▼
                    AI Risk Explanation

```
## 👤 Author

**Ayush Singh**

- 🐙 GitHub: [Ayush Singh](https://https://github.com/haxk1298)
- 💼 LinkedIn: [Ayush Singh](https://linkedin.com/in/heyayush22)
