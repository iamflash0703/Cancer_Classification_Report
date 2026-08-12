"""
================================================================================
app.py — FASTAPI DEPLOYMENT FOR BREAST CANCER SCREENING AI
================================================================================
This file turns your trained model into a LIVE WEBSITE that doctors can use.

WHAT IS FASTAPI?
----------------
FastAPI is a Python framework for building APIs (Application Programming Interfaces).
Think of it like a "waiter" at a restaurant:
  - A doctor (customer) sends a request: "Here are 30 cell measurements"
  - The API (waiter) takes it to the kitchen (your model)
  - The model (chef) cooks the prediction
  - The API (waiter) brings back: "Prediction + Probability + Explanation"

HOW TO RUN LOCALLY:
  1. Open terminal in this folder
  2. Make sure venv is activated
  3. Type: uvicorn app:app --reload
  4. Open browser: http://127.0.0.1:8000/docs
  5. You will see a beautiful page where you can test the API!

HOW TO DEPLOY (Render.com — FREE):
  1. Push this code to GitHub
  2. Go to render.com → Sign up with GitHub
  3. Click "New Web Service" → Connect your repo
  4. Build Command: pip install -r requirements.txt
  5. Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
  6. Click Deploy. Wait 3 minutes. Copy the live URL.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
from fastapi import FastAPI, HTTPException
# FastAPI = The web framework. Makes building APIs super fast.
# HTTPException = For sending error messages (like "400 Bad Request")

from pydantic import BaseModel, Field
# Pydantic = Validates user input automatically.
# Example: If a doctor forgets to send "mean radius", Pydantic says:
# "Hey! You missed a required field!" BEFORE the model crashes.

import joblib          # For LOADING the saved model files
import numpy as np     # For math operations on the input data
import json            # For reading the metadata file
from typing import List, Dict  # For type hints (makes code cleaner)

# ==============================================================================
# CREATE THE FASTAPI APP
# ==============================================================================
app = FastAPI(
    title="Breast Cancer Screening API v2.0",
    description="""
    AI-powered clinical decision support for breast cancer screening.

    **What it does:**
    - Takes 30 cell measurements from a biopsy/mammogram
    - Returns: Cancer probability + Benign probability
    - Returns: Top 3 features that drove the decision (SHAP)
    - Returns: Clinical recommendation based on risk level

    **Why this matters:**
    Threshold is tuned for F2-score (recall-weighted) so it catches
    the maximum number of true cancers, even if it means more false alarms.
    In medicine, missing cancer is far worse than a false alarm.
    """,
    version="2.0.0",
    docs_url="/docs",     # Swagger UI will be at /docs
    redoc_url="/redoc"    # Alternative docs at /redoc
)

# ==============================================================================
# LOAD PRODUCTION ARTIFACTS
# ==============================================================================
# These files were created by breast_cancer_v2_training.py
# We load them ONCE when the server starts (not on every request)
# This makes the API FAST.

try:
    model = joblib.load("breast_cancer_model.pkl")
    # Load the trained model (the "brain")

    scaler = joblib.load("scaler.pkl")
    # Load the scaler (needed to transform NEW patient data the SAME way)

    explainer = joblib.load("shap_explainer.pkl")
    # Load the SHAP explainer (needed to explain predictions)

    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    # Load metadata (threshold, feature names, metrics)

except Exception as e:
    raise RuntimeError(
        f"Failed to load model artifacts. Run training script first. Error: {e}"
    )

# Extract important values from metadata
FEATURE_NAMES = metadata['feature_names']   # List of 30 feature names
THRESHOLD = metadata['threshold']            # The medical-tuned threshold (~0.31)
MODEL_NAME = metadata['model_name']          # Which model won (XGBoost/Random Forest/LogReg)

# ==============================================================================
# DEFINE INPUT SCHEMA (What the doctor sends us)
# ==============================================================================
# Pydantic models define the STRUCTURE of the request.
# Think of it like a form with 30 fields. The doctor MUST fill all 30.

class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,  # "..." means REQUIRED (cannot be empty)
        description="All 30 cell nucleus measurements. Example shows a known malignant case.",
        json_schema_extra={
            "example": {
                "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8,
                "mean area": 1001.0, "mean smoothness": 0.1184, "mean compactness": 0.2776,
                "mean concavity": 0.3001, "mean concave points": 0.1471,
                "mean symmetry": 0.2419, "mean fractal dimension": 0.07871,
                "radius error": 1.095, "texture error": 0.9053, "perimeter error": 8.589,
                "area error": 153.4, "smoothness error": 0.006399,
                "compactness error": 0.04904, "concavity error": 0.05373,
                "concave points error": 0.01587, "symmetry error": 0.03003,
                "fractal dimension error": 0.006193, "worst radius": 25.38,
                "worst texture": 17.33, "worst perimeter": 184.6, "worst area": 2019.0,
                "worst smoothness": 0.1622, "worst compactness": 0.6656,
                "worst concavity": 0.7119, "worst concave points": 0.2654,
                "worst symmetry": 0.4601, "worst fractal dimension": 0.1189
            }
        }
    )

class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Malignant or Benign")
    probability_malignant: float = Field(..., description="Probability of cancer (0-1)")
    probability_benign: float = Field(..., description="Probability of healthy (0-1)")
    threshold_used: float = Field(..., description="Medical-tuned threshold (not 0.5)")
    top_contributing_features: List[Dict] = Field(..., description="SHAP explanations")
    recommendation: str = Field(..., description="Clinical action recommendation")
    model_version: str = Field(..., description="Model identifier")
    sensitivity: str = Field(..., description="Model's cancer detection rate from training")

# ==============================================================================
# API ENDPOINT 1: HOME PAGE (Health Check)
# ==============================================================================
@app.get("/")
def home():
    """
    When someone visits the base URL, show them info about the API.
    This is also useful for monitoring — if this page loads, the API is alive.
    """
    return {
        "message": "Breast Cancer Screening API v2.0 — Clinical Decision Support",
        "model": MODEL_NAME,
        "threshold": THRESHOLD,
        "threshold_rationale": "Tuned for F2-score: catches maximum cancers",
        "sensitivity": f"{metadata['metrics']['recall']*100:.1f}%",
        "docs": "/docs",
        "try_it": "POST /predict with patient features"
    }

# ==============================================================================
# API ENDPOINT 2: THE PREDICTION ENGINE (THE MAIN FEATURE)
# ==============================================================================
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    This is the HEART of the application.

    CLINICAL WORKFLOW:
    1. Lab extracts 30 measurements from biopsy sample
    2. Doctor sends them to this API (via the /predict endpoint)
    3. API returns: probability + top reasons + recommended action
    4. Doctor reviews SHAP explanations before making final call

    WHAT HAPPENS INSIDE:
    1. Validate all 30 features are present
    2. Build a feature array in the EXACT order the model expects
    3. Scale the features (using the same scaler from training)
    4. Get prediction probabilities from the model
    5. Apply the MEDICAL threshold (not 0.5)
    6. Get SHAP explanations for WHY
    7. Generate clinical recommendation
    8. Return everything in a clean JSON response
    """
    try:
        # --- STEP 1: Validate all 30 features are present ---
        missing = [f for f in FEATURE_NAMES if f not in request.features]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing {len(missing)} features: {missing[:5]}..."
            )

        # --- STEP 2: Build feature vector in CORRECT order ---
        # The model expects features in the SAME order as training.
        # We use a list comprehension to ensure correct ordering.
        features_array = np.array([[request.features[f] for f in FEATURE_NAMES]])
        # Result shape: (1, 30) = 1 patient, 30 features

        # --- STEP 3: Scale features ---
        # CRITICAL: We must use the SAME scaler from training.
        # If we don't scale, the model gets confused because it was trained on scaled data.
        features_scaled = scaler.transform(features_array)

        # --- STEP 4: Get prediction probabilities ---
        # predict_proba returns: [[prob_benign, prob_malignant]]
        proba = model.predict_proba(features_scaled)[0]
        prob_malignant = float(proba[1])   # Probability of class 1 (Malignant)
        prob_benign = float(proba[0])      # Probability of class 0 (Benign)

        # --- STEP 5: Apply MEDICAL threshold (NOT default 0.5) ---
        # This is the KEY difference from v1.
        # We use the F2-optimized threshold (~0.31) to catch more cancers.
        is_malignant = prob_malignant >= THRESHOLD

        # --- STEP 6: SHAP Explainability ---
        # We explain THIS specific prediction.
        # The doctor needs to know: "WHY did you say cancer?"
        if MODEL_NAME in ["Random Forest", "XGBoost"]:
            sv = explainer.shap_values(features_scaled)
            if isinstance(sv, list):
                shap_vals = sv[1][0]  # Malignant class explanations for this patient
            else:
                shap_vals = sv[0]
        else:
            shap_vals = explainer.shap_values(features_scaled)[0]

        # Build a list of {feature, shap_value, direction} for ALL features
        feature_impact = [
            {
                "feature": name,
                "shap_value": round(float(val), 6),
                "direction": "increases_risk" if val > 0 else "decreases_risk"
            }
            for name, val in zip(FEATURE_NAMES, shap_vals)
        ]

        # Sort by absolute SHAP value (most impactful first) and take top 3
        feature_impact.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_3 = feature_impact[:3]

        # --- STEP 7: Generate clinical recommendation ---
        # This is what the doctor tells the patient.
        # We make it ACTIONABLE, not just technical.
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

        # --- STEP 8: Return clean JSON response ---
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
        raise  # Re-raise HTTPException (don't catch our own errors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ==============================================================================
# API ENDPOINT 3: HEALTH CHECK (For monitoring tools)
# ==============================================================================
@app.get("/health")
def health():
    """
    Simple health check. Monitoring tools ping this every few minutes.
    If it returns {"status": "healthy"}, the server is alive.
    """
    return {
        "status": "healthy",
        "model_loaded": MODEL_NAME,
        "threshold": THRESHOLD,
        "features_expected": len(FEATURE_NAMES)
    }

# ==============================================================================
# API ENDPOINT 4: FEATURE LIST (For documentation)
# ==============================================================================
@app.get("/features")
def feature_list():
    """
    Returns the list of all 30 required features.
    Useful for frontend developers building the UI.
    """
    return {
        "count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "note": "All 30 features must be provided in the /predict request."
    }

# ==============================================================================
# LOCAL RUN (Only runs when you type: python app.py)
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Starting Breast Cancer Screening API...")
    print(f"Model: {MODEL_NAME}")
    print(f"Threshold: {THRESHOLD}")
    print("Open http://localhost:8000/docs to test the API")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
