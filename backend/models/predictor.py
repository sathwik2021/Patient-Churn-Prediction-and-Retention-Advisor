"""
ML Model Predictor Service — Patient Churn & Retention Advisor 2
"""

import os
import joblib
import numpy as np
import pandas as pd
try:
    import shap
except ImportError:
    shap = None
from typing import Dict, List, Tuple

from schemas.patient import (
    PatientInput,
    FeatureContribution,
    Intervention,
    EngineeredMetrics,
)


class ChurnPredictor:
    """Service to predict patient churn probability, primary churn reason, and retention advice."""

    def __init__(self):
        self.churn_model = None
        self.reason_model = None
        self.columns = None
        self.reason_encoder = None
        self.advice_map = {}
        self.best_thr = 0.37
        self.auc = 0.0
        self._loaded = False

    def load(self, model_dir: str = None):
        """Load pickled artifacts."""
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "ml_model",
            )
        self.churn_model = joblib.load(os.path.join(model_dir, "churn_model.pkl"))
        self.reason_model = joblib.load(os.path.join(model_dir, "reason_model.pkl"))
        self.columns = joblib.load(os.path.join(model_dir, "model_columns.pkl"))
        self.reason_encoder = joblib.load(
            os.path.join(model_dir, "reason_encoder.pkl")
        )
        self.advice_map = joblib.load(os.path.join(model_dir, "advice_map.pkl"))
        thr_path = os.path.join(model_dir, "best_thr.pkl")
        if os.path.exists(thr_path):
            self.best_thr = joblib.load(thr_path)
        auc_path = os.path.join(model_dir, "auc.pkl")
        if os.path.exists(auc_path):
            self.auc = joblib.load(auc_path)
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _build_dataframe(self, patient: PatientInput) -> pd.DataFrame:
        """Convert PatientInput into model feature vector."""
        engagement_score = patient.visits_last_year - patient.missed_appointments
        cost_per_visit = patient.avg_out_of_pocket_cost / (patient.visits_last_year + 1)
        satisfaction_avg = (
            patient.overall_satisfaction
            + patient.wait_time_satisfaction
            + patient.staff_satisfaction
        ) / 3

        row = {
            "Age": patient.age,
            "Tenure_Months": patient.tenure_months,
            "Visits_Last_Year": patient.visits_last_year,
            "Missed_Appointments": patient.missed_appointments,
            "Days_Since_Last_Visit": patient.days_since_last_visit,
            "Overall_Satisfaction": patient.overall_satisfaction,
            "Wait_Time_Satisfaction": patient.wait_time_satisfaction,
            "Staff_Satisfaction": patient.staff_satisfaction,
            "Provider_Rating": patient.provider_rating,
            "Avg_Out_Of_Pocket_Cost": patient.avg_out_of_pocket_cost,
            "Billing_Issues": patient.billing_issues,
            "Portal_Usage": patient.portal_usage,
            "Referrals_Made": patient.referrals_made,
            "Distance_To_Facility_Miles": patient.distance_to_facility,
            "Engagement_Score": engagement_score,
            "Cost_Per_Visit": cost_per_visit,
            "Satisfaction_Avg": satisfaction_avg,
            "Gender": patient.gender,
            "State": patient.state,
            "Specialty": patient.specialty,
            "Insurance_Type": patient.insurance_type,
        }
        df = pd.DataFrame([row])
        df = pd.get_dummies(df)
        df = df.reindex(columns=self.columns, fill_value=0)
        return df

    def _classify_risk(self, probability: float) -> Tuple[str, str]:
        """Classify into High / Medium / Low risk tier (threshold-tuned)."""
        thr = self.best_thr
        if probability >= 0.70:
            return "High", "risk-high"
        elif probability >= thr:
            return "Medium", "risk-medium"
        else:
            return "Low", "risk-low"

    def predict_reason_and_advice(
        self, df_input: pd.DataFrame, patient: PatientInput, probability: float
    ) -> Tuple[str, str]:
        """Predict primary churn reason and map retention advice."""
        if probability < self.best_thr:
            reason = "Not currently at risk (satisfied / engaged patient)"
        else:
            if patient.missed_appointments >= 3:
                reason = "Frequently missed appointments (disengagement)"
            elif patient.billing_issues == 1:
                reason = "Unresolved billing issues"
            elif patient.avg_out_of_pocket_cost >= 1200:
                reason = "High out-of-pocket cost burden"
            elif patient.wait_time_satisfaction <= 2.2:
                reason = "Long appointment wait times"
            elif patient.overall_satisfaction <= 2.2:
                reason = "Low overall satisfaction with care"
            elif patient.staff_satisfaction <= 2.2:
                reason = "Poor front-desk / staff experience"
            elif patient.provider_rating <= 2.5:
                reason = "Low provider rating"
            elif patient.distance_to_facility >= 30.0:
                reason = "Facility located too far from patient"
            elif patient.days_since_last_visit >= 250:
                reason = "Long gap since last visit (lapsed patient)"
            elif patient.visits_last_year <= 1:
                reason = "Low overall visit frequency"
            elif patient.portal_usage == 0:
                reason = "Low patient portal / digital engagement"
            else:
                pred_idx = self.reason_model.predict(df_input)[0]
                reason = str(self.reason_encoder.inverse_transform([pred_idx])[0])

        advice = self.advice_map.get(
            reason,
            "Continue standard engagement: routine check-in reminders and periodic satisfaction surveys.",
        )
        return reason, advice

    def predict_batch(self, df_raw: pd.DataFrame, df_original: pd.DataFrame = None) -> List[Dict]:
        """Fast vectorized batch prediction for entire cohort at once."""
        df = df_raw.copy()

        # Auto-fill missing columns (medians/modes)
        defaults = {
            "Age": 65, "Tenure_Months": 36, "Visits_Last_Year": 2, "Missed_Appointments": 0,
            "Days_Since_Last_Visit": 90, "Overall_Satisfaction": 3.5, "Wait_Time_Satisfaction": 3.5,
            "Staff_Satisfaction": 3.5, "Provider_Rating": 3.5, "Avg_Out_Of_Pocket_Cost": 500,
            "Billing_Issues": 0, "Portal_Usage": 1, "Referrals_Made": 0,
            "Distance_To_Facility_Miles": 15.0, "Gender": "Unknown", "State": "Unknown",
            "Specialty": "Primary Care", "Insurance_Type": "Unknown"
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val
        df.fillna(defaults, inplace=True)

        df["Engagement_Score"] = df["Visits_Last_Year"] - df["Missed_Appointments"]
        df["Cost_Per_Visit"] = df["Avg_Out_Of_Pocket_Cost"] / (df["Visits_Last_Year"] + 1)
        df["Satisfaction_Avg"] = (
            df["Overall_Satisfaction"]
            + df["Wait_Time_Satisfaction"]
            + df["Staff_Satisfaction"]
        ) / 3

        feature_cols = [
            "Age", "Tenure_Months", "Visits_Last_Year", "Missed_Appointments",
            "Days_Since_Last_Visit", "Overall_Satisfaction", "Wait_Time_Satisfaction",
            "Staff_Satisfaction", "Provider_Rating", "Avg_Out_Of_Pocket_Cost",
            "Billing_Issues", "Portal_Usage", "Referrals_Made",
            "Distance_To_Facility_Miles", "Engagement_Score", "Cost_Per_Visit",
            "Satisfaction_Avg", "Gender", "State", "Specialty", "Insurance_Type"
        ]

        X_encoded = pd.get_dummies(df[feature_cols]).reindex(columns=self.columns, fill_value=0)

        probabilities = self.churn_model.predict_proba(X_encoded)[:, 1]
        reason_preds = self.reason_model.predict(X_encoded)
        reason_labels = self.reason_encoder.inverse_transform(reason_preds)

        results = []
        for idx, (_, row) in enumerate(df.iterrows()):
            prob = float(probabilities[idx])
            pct = round(prob * 100, 1)
            risk_level, _ = self._classify_risk(prob)

            if prob < self.best_thr:
                reason = "Not currently at risk (satisfied / engaged patient)"
            else:
                if row.get("Missed_Appointments", 0) >= 3:
                    reason = "Frequently missed appointments (disengagement)"
                elif row.get("Billing_Issues", 0) == 1:
                    reason = "Unresolved billing issues"
                elif row.get("Avg_Out_Of_Pocket_Cost", 0) >= 1200:
                    reason = "High out-of-pocket cost burden"
                elif row.get("Wait_Time_Satisfaction", 5) <= 2.2:
                    reason = "Long appointment wait times"
                elif row.get("Overall_Satisfaction", 5) <= 2.2:
                    reason = "Low overall satisfaction with care"
                elif row.get("Staff_Satisfaction", 5) <= 2.2:
                    reason = "Poor front-desk / staff experience"
                elif row.get("Provider_Rating", 5) <= 2.5:
                    reason = "Low provider rating"
                elif row.get("Distance_To_Facility_Miles", 0) >= 30.0:
                    reason = "Facility located too far from patient"
                elif row.get("Days_Since_Last_Visit", 0) >= 250:
                    reason = "Long gap since last visit (lapsed patient)"
                elif row.get("Visits_Last_Year", 0) <= 1:
                    reason = "Low overall visit frequency"
                elif row.get("Portal_Usage", 1) == 0:
                    reason = "Low patient portal / digital engagement"
                else:
                    reason = str(reason_labels[idx])

            advice = self.advice_map.get(
                reason,
                "Continue standard engagement: routine check-in reminders and periodic satisfaction surveys.",
            )

            patient_id = str(row["PatientID"]) if "PatientID" in row else f"P-{idx+1}"
            
            orig_row = df_original.iloc[idx] if df_original is not None else row

            results.append({
                "index": idx,
                "patient_id": patient_id,
                "probability": round(prob, 4),
                "percentage": pct,
                "risk_level": risk_level,
                "primary_churn_reason": reason,
                "retention_advice": advice,
                "attributes": orig_row.to_dict(),
            })

        return results

    def compute_feature_contributions(self, patient: PatientInput) -> List[FeatureContribution]:
        """Compute relative risk contributions using SHAP."""
        df = self._build_dataframe(patient)
        X_encoded = df.reindex(columns=self.columns, fill_value=0)
        fallback = [
            FeatureContribution(factor="Missed Appointments", risk_impact=0.45),
            FeatureContribution(factor="Days Since Last Visit", risk_impact=0.35),
        ]
        if shap is None:
            return fallback
        try:
            explainer = shap.TreeExplainer(self.churn_model)
            shap_values = explainer.shap_values(X_encoded)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if hasattr(shap_values, "to_numpy"):
                shap_values = shap_values.to_numpy()
            shap_values = np.asarray(shap_values)
            shap_dict = dict(zip(self.columns, shap_values[0]))
            sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
            return [
                FeatureContribution(factor=k.replace("_", " ").title(), risk_impact=round(float(v), 4))
                for k, v in sorted_shap
            ]
        except (ValueError, TypeError, RuntimeError):
            return fallback

    @staticmethod
    def compute_interventions(patient: PatientInput) -> List[Intervention]:
        """Contextual interventions."""
        items: List[Intervention] = []
        if patient.days_since_last_visit > 180:
            items.append(
                Intervention(
                    icon="Outreach",
                    text="Schedule proactive outreach call",
                    priority="high",
                )
            )
        if patient.overall_satisfaction < 2.5:
            items.append(
                Intervention(
                    icon="Advocate", text="Assign patient advocate", priority="high"
                )
            )
        if patient.billing_issues == 1:
            items.append(
                Intervention(
                    icon="Counseling",
                    text="Connect with financial counseling",
                    priority="high",
                )
            )
        if patient.missed_appointments > 3:
            items.append(
                Intervention(
                    icon="Telehealth", text="Offer telehealth options", priority="high"
                )
            )
        if patient.portal_usage == 0:
            items.append(
                Intervention(
                    icon="Portal",
                    text="Promote patient portal enrollment",
                    priority="medium",
                )
            )
        if patient.distance_to_facility > 25:
            items.append(
                Intervention(
                    icon="Location",
                    text="Suggest closer satellite facility",
                    priority="medium",
                )
            )
        if patient.visits_last_year < 2:
            items.append(
                Intervention(
                    icon="Reminders",
                    text="Send preventive care reminders",
                    priority="medium",
                )
            )
        if not items:
            items.append(
                Intervention(
                    icon="Standard",
                    text="Continue standard engagement protocols",
                    priority="low",
                )
            )
        return items

    @staticmethod
    def compute_metrics(patient: PatientInput) -> EngineeredMetrics:
        """Compute metrics."""
        return EngineeredMetrics(
            engagement_score=patient.visits_last_year - patient.missed_appointments,
            satisfaction_avg=round(
                (
                    patient.overall_satisfaction
                    + patient.wait_time_satisfaction
                    + patient.staff_satisfaction
                )
                / 3,
                2,
            ),
            cost_per_visit=round(
                patient.avg_out_of_pocket_cost / (patient.visits_last_year + 1), 2
            ),
            visit_frequency=patient.visits_last_year,
        )

    def predict(self, patient: PatientInput) -> Dict:
        """Run complete prediction pipeline."""
        df = self._build_dataframe(patient)
        probability = float(self.churn_model.predict_proba(df)[0][1])
        percentage = round(probability * 100, 1)
        risk_level, risk_class = self._classify_risk(probability)

        primary_reason, retention_advice = self.predict_reason_and_advice(
            df, patient, probability
        )

        return {
            "probability": round(probability, 4),
            "percentage": percentage,
            "risk_level": risk_level,
            "risk_class": risk_class,
            "primary_churn_reason": primary_reason,
            "retention_advice": retention_advice,
            "metrics": self.compute_metrics(patient),
            "feature_contributions": self.compute_feature_contributions(patient),
            "interventions": self.compute_interventions(patient),
        }


predictor = ChurnPredictor()

