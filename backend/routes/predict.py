"""
API Routes — Patient Churn Prediction
======================================
Prediction, batch upload (CSV + Excel), history, and analytics endpoints.
"""

import io
import json
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from typing import Optional

from models.predictor import predictor
from schemas.patient import (
    PatientInput,
    PredictionResponse,
    BatchPredictionResponse,
    BatchPredictionRow,
    HealthResponse,
)
import database as db
from routes.auth import get_current_user_id

router = APIRouter(prefix="/api", tags=["prediction"])

# Model features that batch uploads should contain (at least 5)
MODEL_FEATURES = [
    "Age", "Gender", "State", "Specialty", "Insurance_Type",
    "Tenure_Months", "Visits_Last_Year", "Missed_Appointments",
    "Days_Since_Last_Visit", "Overall_Satisfaction", "Wait_Time_Satisfaction",
    "Staff_Satisfaction", "Provider_Rating", "Avg_Out_Of_Pocket_Cost",
    "Billing_Issues", "Portal_Usage", "Referrals_Made",
    "Distance_To_Facility_Miles",
]

TARGET_COLUMNS = {"Churned", "Churn", "Target", "Label", "Churn_Reason", "Retention_Advice"}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "unhealthy",
        model_loaded=predictor.is_loaded,
        model_type="XGBoost + Multi-Class Reason Classifier",
        auc=predictor.auc,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_churn(patient: PatientInput, authorization: Optional[str] = Header(None)):
    """Predict churn risk %, primary reason, and retention advice for a single patient."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = predictor.predict(patient)

    # Save to DB if user is authenticated
    user_id = get_current_user_id(authorization)
    if user_id:
        db.save_prediction(
            user_id=user_id,
            patient_data=json.dumps(patient.model_dump()),
            probability=result["probability"],
            risk_level=result["risk_level"],
            primary_reason=result["primary_churn_reason"],
            retention_advice=result["retention_advice"],
        )

    return PredictionResponse(**result)


@router.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    """Batch predict from CSV or Excel file."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    filename = file.filename.lower()
    contents = await file.read()

    # Parse file based on extension
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Only CSV and Excel (.xlsx, .xls) files are supported",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    # Normalize column names: strip whitespace, fix common variations
    df_original = df.copy()
    df.columns = df.columns.str.strip()
    col_mapping = {}
    for col in df.columns:
        normalized = col.replace(" ", "_").replace("-", "_")
        for feat in MODEL_FEATURES:
            if normalized.lower() == feat.lower():
                col_mapping[col] = feat
                break
    df.rename(columns=col_mapping, inplace=True)

    # Targets are optional in inference files and are never model inputs.
    target_columns = [column for column in df.columns if column in TARGET_COLUMNS]
    df.drop(columns=target_columns, inplace=True, errors="ignore")
    if target_columns:
        df_original.drop(columns=target_columns, inplace=True, errors="ignore")

    # Require meaningful overlap with the features used during training.
    matched = [f for f in MODEL_FEATURES if f in df.columns]
    if len(matched) < 5:
        raise HTTPException(
            status_code=400,
            detail="Please upload the correct file",
        )

    batch_results = predictor.predict_batch(df, df_original)

    results = []
    high = medium = low = 0

    for item in batch_results:
        risk = item["risk_level"]
        if risk == "High":
            high += 1
        elif risk == "Medium":
            medium += 1
        else:
            low += 1
        results.append(BatchPredictionRow(**item))

    # Save cohort to DB if authenticated
    user_id = get_current_user_id(authorization)
    if user_id:
        db.save_cohort(user_id, file.filename, len(results), high, medium, low)

    return BatchPredictionResponse(
        total=len(results),
        high_risk=high,
        medium_risk=medium,
        low_risk=low,
        results=results,
    )


@router.get("/history")
async def get_history(authorization: Optional[str] = Header(None)):
    """Get prediction history for authenticated user."""
    user_id = get_current_user_id(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    records = db.get_user_predictions(user_id)
    return {"history": records}


@router.get("/user/analytics")
async def get_analytics(authorization: Optional[str] = Header(None)):
    """Get user analytics dashboard data."""
    user_id = get_current_user_id(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    analytics = db.get_user_analytics(user_id)
    return analytics
