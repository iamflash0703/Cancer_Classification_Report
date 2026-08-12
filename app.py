"""
app.py -- FastAPI Deployment for Breast Cancer Screening AI
Modified: SHAP explainer created dynamically (no need for shap_explainer.pkl)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
from typing import List, Dict

app = FastAPI(
    title="Breast Cancer Screening API v2.0",
    description="AI-powered clinical decision support for breast cancer screening.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Load production artifacts
try:
    model = joblib.load("breast_cancer_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts. Run training script first. Error: {e}")

FEATURE_NAMES = metadata['feature_names']
THRESHOLD = metadata['threshold']
MODEL_NAME = metadata['model_name']

# ============================================================================
# FIX: Create SHAP explainer dynamically from model (no .pkl needed!)
# ============================================================================
try:
    import shap
    # TreeExplainer works for Random Forest and XGBoost
    # For Logistic Regression, we use a simpler approach
    if MODEL_NAME in ["Random Forest", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
    else:
        # For Logistic Regression, create a simple KernelExplainer
        # Using a small sample of data for background
        sample_data = np.zeros((1, len(FEATURE_NAMES)))  # dummy sample
        explainer = shap.KernelExplainer(model.predict_proba, sample_data)
except Exception as e:
    print(f"Warning: Could not create SHAP explainer: {e}")
    explainer = None

# Input schema
class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="All 30 cell nucleus measurements")

class PredictResponse(BaseModel):
    prediction: str
    probability_malignant: float
    probability_benign: float
    threshold_used: float
    top_contributing_features: List[Dict]
    recommendation: str
    model_version: str
    sensitivity: str

@app.get("/")
def home():
    return {
        "message": "Breast Cancer Screening API v2.0",
        "model": MODEL_NAME,
        "threshold": THRESHOLD,
        "sensitivity": f"{metadata['metrics']['recall']*100:.1f}%",
        "docs": "/docs"
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # Validate features
        missing = [f for f in FEATURE_NAMES if f not in request.features]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing {len(missing)} features: {missing[:5]}...")

        # Build and scale features
        features_array = np.array([[request.features[f] for f in FEATURE_NAMES]])
        features_scaled = scaler.transform(features_array)

        # Predict
        proba = model.predict_proba(features_scaled)[0]
        prob_malignant = float(proba[1])
        prob_benign = float(proba[0])
        is_malignant = prob_malignant >= THRESHOLD

        # SHAP explainability
        top_3 = []
        if explainer is not None and MODEL_NAME in ["Random Forest", "XGBoost"]:
            try:
                sv = explainer.shap_values(features_scaled)
                if isinstance(sv, list):
                    shap_vals = sv[1][0]
                else:
                    shap_vals = sv[0]

                feature_impact = [
                    {"feature": name, "shap_value": round(float(val), 6),
                     "direction": "increases_risk" if val > 0 else "decreases_risk"}
                    for name, val in zip(FEATURE_NAMES, shap_vals)
                ]
                feature_impact.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
                top_3 = feature_impact[:3]
            except Exception:
                top_3 = []

        # Recommendation
        if is_malignant:
            if prob_malignant > 0.85:
                rec = "HIGH RISK: Immediate biopsy and oncologist consultation strongly recommended. Do not delay."
            else:
                rec = "ELEVATED RISK: Schedule diagnostic mammogram and biopsy within 5-7 days."
        else:
            if prob_benign > 0.90:
                rec = "LOW RISK: Continue routine annual screening. No immediate action required."
            else:
                rec = "LOW-MODERATE RISK: Repeat screening in 6 months. Monitor for changes."

        return PredictResponse(
            prediction="Malignant" if is_malignant else "Benign",
            probability_malignant=round(prob_malignant, 4),
            probability_benign=round(prob_benign, 4),
            threshold_used=round(THRESHOLD, 4),
            top_contributing_features=top_3,
            recommendation=rec,
            model_version=f"v2.0-{MODEL_NAME.lower().replace(' ','-')}-f2-optimized",
            sensitivity=f"{metadata['metrics']['recall']*100:.1f}%"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL_NAME, "threshold": THRESHOLD}

@app.get("/features")
def feature_list():
    return {"count": len(FEATURE_NAMES), "features": FEATURE_NAMES}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
