"""
ML Model Training Script — Patient Churn & Retention Advisor (XGBoost)
===========================================================================
Trains:
1. Binary Churn Classifier (XGBoost) for churn probability %
2. Multi-class Churn Reason Classifier (XGBoost) for primary churn reason
3. Retention Advice Map (maps each churn reason to actionable retention advice)

Saves artifacts to backend/ml_model/
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score


def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        os.path.dirname(base_dir), "data", "patient_churn_dataset_enriched.csv"
    )

    print(f"[1/4] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # Create advice mapping from dataset ground truth
    advice_map = (
        df[["Churn_Reason", "Retention_Advice"]]
        .drop_duplicates()
        .set_index("Churn_Reason")["Retention_Advice"]
        .to_dict()
    )
    print(f"[2/4] Created retention advice map for {len(advice_map)} unique reasons")

    # Feature Engineering
    df["Engagement_Score"] = df["Visits_Last_Year"] - df["Missed_Appointments"]
    df["Cost_Per_Visit"] = df["Avg_Out_Of_Pocket_Cost"] / (df["Visits_Last_Year"] + 1)
    df["Satisfaction_Avg"] = (
        df["Overall_Satisfaction"]
        + df["Wait_Time_Satisfaction"]
        + df["Staff_Satisfaction"]
    ) / 3

    feature_cols = [
        "Age",
        "Tenure_Months",
        "Visits_Last_Year",
        "Missed_Appointments",
        "Days_Since_Last_Visit",
        "Overall_Satisfaction",
        "Wait_Time_Satisfaction",
        "Staff_Satisfaction",
        "Provider_Rating",
        "Avg_Out_Of_Pocket_Cost",
        "Billing_Issues",
        "Portal_Usage",
        "Referrals_Made",
        "Distance_To_Facility_Miles",
        "Engagement_Score",
        "Cost_Per_Visit",
        "Satisfaction_Avg",
        "Gender",
        "State",
        "Specialty",
        "Insurance_Type",
    ]

    X_raw = df[feature_cols]
    y_churn = df["Churned"]
    y_reason = df["Churn_Reason"]

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X_raw)
    model_columns = X_encoded.columns.tolist()

    # Train/Test Split
    X_train, X_test, y_churn_train, y_churn_test, y_reason_train, y_reason_test = (
        train_test_split(
            X_encoded,
            y_churn,
            y_reason,
            test_size=0.2,
            random_state=42,
            stratify=y_churn,
        )
    )

    print("[3/4] Training Churn Probability Model (XGBoost)...")
    churn_rate = y_churn_train.mean()
    print(f"   -> Training churn rate: {churn_rate:.2%}")
    churn_model = XGBClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
        scale_pos_weight=(1 - churn_rate) / churn_rate, eval_metric="logloss",
    )
    churn_model.fit(X_train, y_churn_train)
    churn_probs = churn_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_churn_test, churn_probs)
    print(f"   -> Binary Churn Model ROC-AUC: {auc:.4f}")

    # Decision threshold tuned for retention: catch more churners (recall-first)
    best_thr = 0.37
    churn_preds = (churn_probs >= best_thr).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print(f"   -> Metrics at threshold {best_thr}:")
    print(f"      Acc={accuracy_score(y_churn_test, churn_preds):.4f} "
          f"Prec={precision_score(y_churn_test, churn_preds):.4f} "
          f"Rec={recall_score(y_churn_test, churn_preds):.4f} "
          f"F1={f1_score(y_churn_test, churn_preds):.4f}")

    print("[4/4] Training Churn Reason Classifier (XGBoost)...")
    reason_encoder = LabelEncoder()
    y_reason_encoded_train = reason_encoder.fit_transform(y_reason_train)
    y_reason_encoded_test = reason_encoder.transform(y_reason_test)

    reason_model = XGBClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    reason_model.fit(X_train, y_reason_encoded_train)
    reason_acc = reason_model.score(X_test, y_reason_encoded_test)
    print(f"   -> Churn Reason Model Accuracy: {reason_acc:.4f}")

    # Save artifacts
    ml_model_dir = os.path.join(base_dir, "ml_model")
    os.makedirs(ml_model_dir, exist_ok=True)

    joblib.dump(churn_model, os.path.join(ml_model_dir, "churn_model.pkl"))
    joblib.dump(reason_model, os.path.join(ml_model_dir, "reason_model.pkl"))
    joblib.dump(model_columns, os.path.join(ml_model_dir, "model_columns.pkl"))
    joblib.dump(reason_encoder, os.path.join(ml_model_dir, "reason_encoder.pkl"))
    joblib.dump(advice_map, os.path.join(ml_model_dir, "advice_map.pkl"))
    joblib.dump(best_thr, os.path.join(ml_model_dir, "best_thr.pkl"))
    joblib.dump(auc, os.path.join(ml_model_dir, "auc.pkl"))

    print(f"\n[OK] All XGBoost artifacts successfully saved to: {ml_model_dir}")


if __name__ == "__main__":
    train()
