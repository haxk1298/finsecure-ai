from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_groq import ChatGroq


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "xboost_model.pkl"
STATIC_DIR = BASE_DIR / "static"


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(title="AI Fraud Detection System")


# ============================================================
# CORS configuration
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Static files
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# ============================================================
# Load ML model
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# Groq LLM configuration
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set.")


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)


# ============================================================
# Model features
# ============================================================

features = [
    "amount",
    "log_amount",
    "recency_hours",
    "txn_count_24h",
    "is_dest_new",
    "hours_day",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]


# ============================================================
# Request model
# ============================================================

class Transaction(BaseModel):
    amount: float
    recency_hours: float
    txn_count_24h: int
    is_dest_new: int
    hours_day: int

    oldbalanceOrg: float
    newbalanceOrig: float

    oldbalanceDest: float
    newbalanceDest: float

    type_CASH_IN: int
    type_CASH_OUT: int
    type_DEBIT: int
    type_PAYMENT: int
    type_TRANSFER: int


# ============================================================
# Response model
# ============================================================

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    decision: str
    risk_level: str
    summary: str


# ============================================================
# Home route
# ============================================================

@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


# ============================================================
# Fraud prediction endpoint
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction):

    # --------------------------------------------------------
    # Convert request to dictionary
    # --------------------------------------------------------

    txn_dict = transaction.model_dump()

    # Calculate log-transformed amount
    txn_dict["log_amount"] = np.log1p(txn_dict["amount"])


    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame([txn_dict])

    # Keep the exact feature order expected by the model
    X = df[features]


    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    fraud_prob = float(
        model.predict_proba(X)[0, 1]
    )

    is_fraud = fraud_prob >= 0.8


    # --------------------------------------------------------
    # Determine risk level
    # --------------------------------------------------------

    if fraud_prob >= 0.8:
        risk_level = "HIGH"

    elif fraud_prob >= 0.5:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"


    # --------------------------------------------------------
    # Determine transaction type
    # --------------------------------------------------------

    if txn_dict["type_TRANSFER"] == 1:
        txn_type = "TRANSFER"

    elif txn_dict["type_CASH_OUT"] == 1:
        txn_type = "CASH_OUT"

    elif txn_dict["type_PAYMENT"] == 1:
        txn_type = "PAYMENT"

    elif txn_dict["type_CASH_IN"] == 1:
        txn_type = "CASH_IN"

    else:
        txn_type = "DEBIT"


    # --------------------------------------------------------
    # Create prompt for Groq
    # --------------------------------------------------------

    prompt = f"""
You are a fraud detection analyst.
Analyze this transaction and provide a clear, professional summary.

ANALYSIS RESULTS:

- Fraud Probability: {fraud_prob * 100:.2f}%
- Decision: {"FRAUD DETECTED" if is_fraud else "LEGITIMATE TRANSACTION"}
- Risk Level: {risk_level}

TRANSACTION DETAILS:

- Amount: ${txn_dict["amount"]:,.2f}
- Type: {txn_type}

- Account Balance Before: ${txn_dict["oldbalanceOrg"]:,.2f}
- Account Balance After: ${txn_dict["newbalanceOrig"]:,.2f}

- Destination Balance Before: ${txn_dict["oldbalanceDest"]:,.2f}
- Destination Balance After: ${txn_dict["newbalanceDest"]:,.2f}

- New Destination: {"Yes" if txn_dict["is_dest_new"] == 1 else "No"}
- Hours Since Last Transaction: {txn_dict["recency_hours"]}
- Transactions in Last 24h: {txn_dict["txn_count_24h"]}

Provide a 2-3 sentence professional summary explaining why this
transaction is {"flagged as fraud" if is_fraud else "considered legitimate"}.

Focus on the key risk indicators.
"""


    # --------------------------------------------------------
    # Generate AI explanation
    # --------------------------------------------------------

    try:

        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a financial fraud detection expert."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        summary = response.content.strip()

    except Exception as e:

        summary = (
            "Unable to generate AI summary. "
            f"Error: {str(e)}"
        )


    # --------------------------------------------------------
    # Return prediction
    # --------------------------------------------------------

    return {
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud": is_fraud,
        "decision": "FRAUD" if is_fraud else "LEGITIMATE",
        "risk_level": risk_level,
        "summary": summary
    }