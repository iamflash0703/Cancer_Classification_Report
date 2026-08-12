"""
================================================================================
BREAST CANCER CLASSIFICATION v2 — PRODUCTION READY
================================================================================
Dataset: Wisconsin Breast Cancer (REAL clinical data, 569 patients, 30 features)
Goal: Build a deployable clinical AI that predicts if a tumor is cancerous

WHAT IS NEW IN v2 (compared to your old v1):
  v1 had: Logistic Regression vs Decision Tree, basic train-test split, 98% accuracy
  v2 has: XGBoost + Random Forest + LogReg, 5-Fold CV, F2-score tuning, SHAP, deployed API

AUTHOR: Shovit Nayak
"""

# ==============================================================================
# STEP 0: IMPORT ALL LIBRARIES
# ==============================================================================
# Think of imports like "opening your toolbox" before starting work.
# We are telling Python: "Hey, I need these tools. Go get them."

import pandas as pd           # pd = "pandas". For handling tables/data like Excel
import numpy as np            # np = "numpy". For math, numbers, arrays
import matplotlib.pyplot as plt  # For drawing graphs and charts
import seaborn as sns         # sns = "seaborn". Makes prettier charts than matplotlib
import joblib                 # For SAVING our trained model (like Save Game)
import json                   # For saving text data (metrics, settings) in a file
import warnings               # For hiding annoying red warning messages
warnings.filterwarnings('ignore')  # "Hey Python, don't show warnings. I know what I'm doing."

# ------------------------------------------------------------------------------
# Machine Learning tools from scikit-learn (sklearn)
# scikit-learn is the MOST POPULAR ML library in Python. It's free and powerful.
# ------------------------------------------------------------------------------
from sklearn.datasets import load_breast_cancer
# "load_breast_cancer" = This function brings the Wisconsin Breast Cancer dataset
# It's BUILT-IN. You don't need to download anything. It has 569 real patients.

from sklearn.model_selection import StratifiedKFold, cross_val_predict
# "StratifiedKFold" = A SMART way to split data. It keeps the same % of cancer cases
# in EVERY fold. So your results are HONEST, not lucky.
# "cross_val_predict" = Runs your model on ALL data, but tests on different parts each time.

from sklearn.preprocessing import StandardScaler
# "StandardScaler" = Makes all numbers the SAME SIZE.
# Example: "radius" might be 20, "smoothness" might be 0.1. The model gets confused.
# StandardScaler converts everything to "mean = 0, standard deviation = 1"
# This is CRITICAL for models like Logistic Regression and SVM.

from sklearn.linear_model import LogisticRegression
# Your old friend from v1. Simple, fast, interpretable baseline model.

from sklearn.ensemble import RandomForestClassifier
# "Random Forest" = Builds 200 mini Decision Trees and takes a VOTE.
# Much better than a single Decision Tree. Less overfitting.

from sklearn.metrics import (
    accuracy_score,      # "How many did I get right?" (can be misleading)
    precision_score,     # "Of all I predicted as cancer, how many ACTUALLY had cancer?"
    recall_score,        # "Of all who ACTUALLY had cancer, how many did I catch?" <- MOST IMPORTANT
    f1_score,            # Balance between Precision and Recall
    fbeta_score,         # Like F1, but we can WEIGHT Recall higher (F2-score)
    roc_auc_score,       # "How well can I separate cancer vs healthy?" (0.5 = random, 1.0 = perfect)
    average_precision_score,  # Another way to measure separation quality
    confusion_matrix,    # The famous 2x2 table: True Positives, False Positives, etc.
    roc_curve,           # For drawing the ROC curve (True Positive Rate vs False Positive Rate)
    precision_recall_curve  # For drawing Precision-Recall curve (used for threshold tuning)
)

# ------------------------------------------------------------------------------
# XGBoost = eXtreme Gradient Boosting. This is THE secret weapon.
# It's a type of "ensemble" model that corrects its own mistakes.
# XGBoost wins 70% of Kaggle competitions. It's that good.
# ------------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("\n>>> Please install XGBoost first: pip install xgboost <<<")

# ------------------------------------------------------------------------------
# SHAP = SHapley Additive exPlanations. This makes your model EXPLAINABLE.
# In medicine, doctors won't trust a black box. SHAP tells them:
# "You predicted cancer because the tumor radius was 25mm (very large)"
# ------------------------------------------------------------------------------
try:
    import shap
except ImportError:
    raise ImportError("\n>>> Please install SHAP: pip install shap <<<")

# ==============================================================================
# SETUP: Make our graphs look good
# ==============================================================================
sns.set_style("whitegrid")   # "whitegrid" = white background with light grid lines
plt.rcParams["figure.dpi"] = 150  # "dpi" = dots per inch. Higher = sharper image.
np.random.seed(42)           # "seed" = Makes random numbers PREDICTABLE.
# Why seed? So every time you run this code, you get the SAME results.
# This is important for SCIENCE and for INTERVIEWS.

# ==============================================================================
# STEP 1: LOAD THE REAL CLINICAL DATA
# ==============================================================================
# The Wisconsin Breast Cancer dataset is FAMOUS in ML.
# It contains 569 patients who had breast tumors.
# Doctors took a "fine needle aspirate" (FNA) — a small needle sample —
# and measured 30 features of the cell nuclei under a microscope.
# Then they checked: Was it Malignant (cancer) or Benign (harmless)?

print("=" * 70)
print("STEP 1: Loading Wisconsin Breast Cancer Dataset")
print("=" * 70)

data = load_breast_cancer()
# "data" is like a dictionary. It has:
#   data.data = the 30 features (numbers)
#   data.target = 0 or 1 (0 = Malignant, 1 = Benign in original)
#   data.feature_names = names of the 30 columns
#   data.target_names = ['malignant', 'benign']

# Convert to Pandas DataFrame (like an Excel sheet in Python)
X = pd.DataFrame(data.data, columns=data.feature_names)
# X = "Input features". The 30 measurements we use to PREDICT.
# Example columns: 'mean radius', 'mean texture', 'mean perimeter', etc.

# IMPORTANT FIX FOR v2:
# In the original dataset: 0 = Malignant, 1 = Benign
# But in MEDICINE, we care about detecting the DANGEROUS thing.
# So we FLIP it: 1 = Malignant (the thing we MUST catch), 0 = Benign
y = pd.Series(1 - data.target, name="malignant")
# "1 - data.target" means: if original was 0 (malignant), new is 1. If original was 1 (benign), new is 0.

print(f"Total patients (samples): {X.shape[0]}")
print(f"Features per patient: {X.shape[1]}")
print(f"Healthy (Benign): {(y==0).sum()}")
print(f"Cancer (Malignant): {(y==1).sum()}")
print(f"Cancer rate: {y.mean()*100:.1f}%\n")
# Note: ~37% of patients have cancer. This is "imbalanced" but not extreme.

# ==============================================================================
# STEP 1b: QUICK EDA (Exploratory Data Analysis)
# ==============================================================================
# Before building models, we MUST understand our data.
# Let's look at the TOP 5 features and how they relate to cancer.

top5 = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean concavity']
# These are the 5 most "obvious" features. Even doctors look at these first.

plt.figure(figsize=(10, 8))
# Create a figure that is 10 inches wide, 8 inches tall

corr = pd.concat([X[top5], y], axis=1).corr()
# "corr()" = correlation. It tells us: "When X goes up, does Y go up too?"
# +1.0 = perfect positive (when radius increases, cancer increases)
# -1.0 = perfect negative
# 0.0 = no relationship

mask = np.triu(np.ones_like(corr, dtype=bool))
# "triu" = upper triangle. We only show the bottom half of the correlation matrix.
# Why? Because the top half is just a MIRROR of the bottom half. No new info.

sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True)
# Draw a heatmap (colorful grid):
#   annot=True = write the numbers inside each box
#   fmt='.2f' = show 2 decimal places (0.85, not 0.85321)
#   cmap='RdBu_r' = Red-Blue color map. Red = positive, Blue = negative
#   center=0 = White color at 0 (no correlation)

plt.title("Clinical Feature Correlation + Malignancy")
plt.tight_layout()  # "tight_layout" = adjust spacing so nothing gets cut off
plt.savefig("eda_correlation_heatmap.png")
# "savefig" = SAVE this image as a PNG file. You can put this on your resume!
plt.close()
print("[✓] Saved: eda_correlation_heatmap.png")

# ==============================================================================
# STEP 2: STRATIFIED 5-FOLD CROSS-VALIDATION
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Stratified 5-Fold Cross-Validation")
print("=" * 70)
print("""
WHY CROSS-VALIDATION? (The most important concept in this code)
-----------------------------------------------------------------
In your v1, you did: train_test_split (one split, 80% train, 20% test)
Problem: If you're LUCKY, your test set is easy. If you're UNLUCKY, it's hard.
You might get 98% one time and 85% another time. That's NOT reliable.

CROSS-VALIDATION FIXES THIS:
  1. Split data into 5 parts (folds)
  2. Train on 4 parts, test on 1 part. Record score.
  3. Train on a DIFFERENT 4 parts, test on the remaining 1. Record score.
  4. Do this 5 times. Every patient gets to be in the "test set" exactly once.
  5. Average the 5 scores. This is your HONEST score.

STRATIFIED means: Each fold keeps the SAME % of cancer patients.
  If 37% of patients have cancer, then EVERY fold has ~37% cancer patients.
  This is CRITICAL for medical data. Otherwise one fold might have 50% cancer
  and another might have 10%. That would give WRONG results.
""")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# n_splits=5 = 5 folds
# shuffle=True = Randomly shuffle before splitting (so order doesn't matter)
# random_state=42 = Same shuffle every time (reproducible)

scaler = StandardScaler()
# Create our "scaler" object. It will learn the mean and standard deviation
# of each feature, then transform everything to the same scale.

X_scaled = scaler.fit_transform(X)
# "fit" = Learn the mean and std of each column from the FULL data.
# "transform" = Apply the scaling: (value - mean) / std
# Result: Every column now has mean=0 and std=1

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
# Convert back to DataFrame so we can use column names later (for SHAP)

# ==============================================================================
# STEP 3: DEFINE AND TRAIN 3 MODELS
# ==============================================================================
print("=" * 70)
print("STEP 3: Training 3 Models with Cross-Validation")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000,           # Allow up to 5000 iterations to converge
        class_weight='balanced', # IMPORTANT: Give MORE weight to the minority class (cancer)
                                 # Because cancer patients are only 37%, the model might ignore them.
                                 # 'balanced' fixes this automatically.
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,        # Build 200 Decision Trees and vote
        max_depth=6,             # Don't let trees grow deeper than 6 levels (prevents overfitting)
        min_samples_leaf=4,      # Each leaf must have at least 4 patients (prevents overfitting)
        class_weight='balanced', # Same as above: weight cancer cases more
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,        # 200 boosting rounds
        max_depth=4,             # Shallow trees (prevents overfitting)
        learning_rate=0.05,      # Small steps = more careful learning = better generalization
        subsample=0.8,           # Use 80% of data for each tree (prevents overfitting)
        colsample_bytree=0.8,    # Use 80% of features for each tree (prevents overfitting)
        scale_pos_weight=(y==0).sum()/(y==1).sum(),
        # CRITICAL for imbalanced data: Tell XGBoost "healthy patients are ~1.7x more common"
        # So pay MORE attention to the rare cancer cases.
        eval_metric='logloss',   # How XGBoost measures error internally
        random_state=42
    )
}

# ------------------------------------------------------------------------------
# Dictionary to store results for all 3 models
# ------------------------------------------------------------------------------
cv_results = {}

for name, model in models.items():
    print(f"\n>>> Training: {name}")

    # cross_val_predict gives us the PREDICTIONS for every patient,
    # but each patient was tested when it was in the "test fold"
    # This is UNBIASED — the model never saw the patient during training.
    y_proba = cross_val_predict(model, X_scaled, y, cv=skf, method='predict_proba')[:, 1]
    # method='predict_proba' = Give us PROBABILITIES, not just 0/1
    # [:, 1] = We only want the probability of class 1 (Malignant)

    # Default prediction: If probability >= 0.5, predict cancer
    y_pred_default = (y_proba >= 0.5).astype(int)

    # Store everything in our dictionary
    cv_results[name] = {
        'model': model,
        'y_proba': y_proba,  # The probability scores (we need these for threshold tuning)
        'accuracy': accuracy_score(y, y_pred_default),
        'precision': precision_score(y, y_pred_default, zero_division=0),
        'recall': recall_score(y, y_pred_default, zero_division=0),
        'f1': f1_score(y, y_pred_default, zero_division=0),
        'f2': fbeta_score(y, y_pred_default, beta=2, zero_division=0),
        'roc_auc': roc_auc_score(y, y_proba),
        'pr_auc': average_precision_score(y, y_proba)
    }

    # Print the metrics
    print(f"   Accuracy:  {cv_results[name]['accuracy']:.4f}")
    print(f"   Precision: {cv_results[name]['precision']:.4f}  (of predicted cancer, how many real?)")
    print(f"   Recall:    {cv_results[name]['recall']:.4f}  (of real cancer, how many caught?)")
    print(f"   F1:        {cv_results[name]['f1']:.4f}")
    print(f"   F2:        {cv_results[name]['f2']:.4f}  <-- MEDICAL PRIORITY (Recall weighted 2x)")
    print(f"   ROC-AUC:   {cv_results[name]['roc_auc']:.4f}")
    print(f"   PR-AUC:    {cv_results[name]['pr_auc']:.4f}")

# ==============================================================================
# STEP 4: THRESHOLD TUNING (The Medical Secret Weapon)
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Threshold Tuning for Maximum F2-Score")
print("=" * 70)
print("""
WHY ARE WE DOING THIS?
-----------------------
Every model outputs a PROBABILITY (0.0 to 1.0):
  "This patient has 0.72 probability of cancer"

By DEFAULT, we say: "If probability >= 0.5, call it cancer."
But 0.5 is just a GUESS. It treats "missing cancer" and "false alarm" equally.

In MEDICINE, they are NOT equal:
  Missing cancer (False Negative) = Patient dies. Cost = ₹10,00,000+ and a life.
  False alarm (False Positive) = Unnecessary biopsy. Cost = ₹500 and some anxiety.

So we should LOWER the threshold:
  "If probability >= 0.3, call it cancer"
  This catches MORE real cancers (higher Recall) but also creates more false alarms.

F2-SCORE is the METRIC that weights Recall 2x higher than Precision.
We test EVERY threshold from 0.01 to 0.99 and pick the one that gives MAXIMUM F2.
""")

best_thresholds = {}  # Store the best threshold for each model

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Create 3 side-by-side plots (one for each model)

for idx, (name, res) in enumerate(cv_results.items()):
    y_proba = res['y_proba']

    # precision_recall_curve gives us 3 things:
    #   precisions = precision at each threshold
    #   recalls = recall at each threshold
    #   thresholds = the actual threshold values tested
    precisions, recalls, thresholds = precision_recall_curve(y, y_proba)

    # Calculate F2-score at EVERY threshold
    # F2 = (1 + 2^2) * (Precision * Recall) / ((2^2 * Precision) + Recall)
    # Simplified: F2 = 5PR / (4P + R)
    f2_scores = []
    for p, r in zip(precisions, recalls):
        if p + r == 0:
            f2_scores.append(0)
        else:
            f2_scores.append((5 * p * r) / ((4 * p) + r))

    # Find the threshold that gives MAXIMUM F2
    best_idx = np.argmax(f2_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_thresholds[name] = best_thresh

    # Draw the plot for this model
    ax = axes[idx]
    ax.plot(thresholds, f2_scores[:-1], color='#2ECC71', linewidth=2, label='F2-Score')
    # f2_scores[:-1] because precision_recall_curve returns one extra value

    ax.axvline(best_thresh, color='#E74C3C', linestyle='--',
               label=f'Best Threshold = {best_thresh:.3f}')
    # Draw a red dashed line at the best threshold

    ax.set_title(f"{name}\nMax F2 = {f2_scores[best_idx]:.4f}")
    ax.set_xlabel("Threshold (Probability Cutoff)")
    ax.set_ylabel("F2-Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Medical Threshold Tuning: Maximizing Cancer Detection (F2-Score)",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("threshold_tuning_f2.png")
plt.close()
print("[✓] Saved: threshold_tuning_f2.png")

# ==============================================================================
# STEP 4b: RE-EVALUATE AT OPTIMAL THRESHOLDS
# ==============================================================================
print("\n>>> Performance at OPTIMAL Threshold (NOT default 0.5):")
optimal_results = {}

for name, res in cv_results.items():
    thresh = best_thresholds[name]
    y_pred_opt = (res['y_proba'] >= thresh).astype(int)
    # "astype(int)" converts True/False to 1/0

    optimal_results[name] = {
        'threshold': thresh,
        'accuracy': accuracy_score(y, y_pred_opt),
        'precision': precision_score(y, y_pred_opt, zero_division=0),
        'recall': recall_score(y, y_pred_opt, zero_division=0),
        'f1': f1_score(y, y_pred_opt, zero_division=0),
        'f2': fbeta_score(y, y_pred_opt, beta=2, zero_division=0)
    }

    print(f"\n{name} (threshold = {thresh:.3f}):")
    print(f"   Accuracy:  {optimal_results[name]['accuracy']:.4f}")
    print(f"   Precision: {optimal_results[name]['precision']:.4f}")
    print(f"   Recall:    {optimal_results[name]['recall']:.4f}  <-- Cancer catch rate")
    print(f"   F1:        {optimal_results[name]['f1']:.4f}")
    print(f"   F2:        {optimal_results[name]['f2']:.4f}  <-- Our target metric")

# ==============================================================================
# STEP 5: SELECT BEST MODEL AND TRAIN ON FULL DATA
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Final Model Selection")
print("=" * 70)

# Pick the model with the HIGHEST F2-score at optimal threshold
best_model_name = max(optimal_results, key=lambda k: optimal_results[k]['f2'])
print(f"🏆 WINNER: {best_model_name}")
print(f"   Optimal Threshold: {optimal_results[best_model_name]['threshold']:.3f}")
print(f"   Best F2-Score: {optimal_results[best_model_name]['f2']:.4f}\n")

best_model = models[best_model_name]
best_thresh = optimal_results[best_model_name]['threshold']

# Train on FULL dataset for production
# In real life, you train on ALL available data before deployment.
# The cross-validation was just to PICK the best model and threshold.
best_model.fit(X_scaled, y)

# ==============================================================================
# STEP 6: SHAP EXPLAINABILITY (Clinical Transparency)
# ==============================================================================
print("=" * 70)
print("STEP 6: SHAP Explainability")
print("=" * 70)
print("""
WHAT IS SHAP?
-------------
SHAP tells us: "For THIS specific patient, which features pushed the prediction
 toward cancer, and which pushed it toward healthy?"

Example output for one patient:
  worst perimeter (+0.82)  -> INCREASES cancer risk (large tumor)
  mean smoothness (-0.15)  -> DECREASES cancer risk (smooth cells are good)
  mean concavity (+0.65)   -> INCREASES cancer risk (irregular shape)

Doctors need this. Patients need this. Regulators need this.
Without SHAP, your model is a "black box" that nobody trusts.
""")

# TreeSHAP is FAST and designed for tree-based models (Random Forest, XGBoost)
if best_model_name in ["Random Forest", "XGBoost"]:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_scaled_df)
    # shap_values is a list for binary classification: [shap_for_class_0, shap_for_class_1]
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]  # We want explanations for class 1 (Malignant)
    else:
        shap_values_pos = shap_values
else:
    # For Logistic Regression, we use KernelExplainer (slower but works for any model)
    explainer = shap.KernelExplainer(best_model.predict, shap.sample(X_scaled_df, 50))
    shap_values_pos = explainer.shap_values(X_scaled_df)

# Draw the SHAP summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_pos, X_scaled_df, feature_names=X.columns, show=False)
# show=False = Don't display immediately. We want to save it first.
plt.title(f"SHAP: Features Driving Malignancy Predictions ({best_model_name})")
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()
print("[✓] Saved: shap_summary.png")

# Print top 10 features by average SHAP value
mean_shap = np.abs(shap_values_pos).mean(axis=0)
shap_importance = pd.DataFrame({
    'feature': X.columns,
    'mean_shap': mean_shap
}).sort_values('mean_shap', ascending=False)

print("\nTop 10 Features Driving Malignancy (by SHAP importance):")
print(shap_importance.head(10).to_string(index=False))

# ==============================================================================
# STEP 7: FINAL VISUALIZATIONS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Final Visualizations")
print("=" * 70)

# Get final predictions using the OPTIMAL threshold
y_final_proba = best_model.predict_proba(X_scaled)[:, 1]
y_final_pred = (y_final_proba >= best_thresh).astype(int)

# --- Confusion Matrix ---
cm = confusion_matrix(y, y_final_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='mako',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title(f"Confusion Matrix — {best_model_name} (threshold={best_thresh:.3f})")
plt.xlabel("Predicted")   # What the model SAID
plt.ylabel("Actual")      # What was REALLY true
plt.tight_layout()
plt.savefig("confusion_matrix_final.png")
plt.close()
print("[✓] Saved: confusion_matrix_final.png")

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y, y_final_proba)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='#2ECC71', linewidth=2,
         label=f'ROC Curve (AUC = {roc_auc_score(y, y_final_proba):.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guessing')
plt.xlabel('False Positive Rate (Healthy people flagged as cancer)')
plt.ylabel('True Positive Rate / Recall (Cancer people correctly caught)')
plt.title('ROC Curve — Final Production Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()
print("[✓] Saved: roc_curve.png")

# ==============================================================================
# STEP 8: SAVE PRODUCTION ARTIFACTS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 8: Saving Production Artifacts")
print("=" * 70)

# joblib = Python's way of saving objects to disk
# Think of it like "Save Game" in a video game.
# We save the model, scaler, and explainer so we can LOAD them later in our API.

joblib.dump(best_model, "breast_cancer_model.pkl")
# "breast_cancer_model.pkl" = The trained model file. This is the BRAIN.

joblib.dump(scaler, "scaler.pkl")
# "scaler.pkl" = The scaler object. We need this to transform NEW patient data
# the SAME WAY we transformed training data.

joblib.dump(explainer, "shap_explainer.pkl")
# "shap_explainer.pkl" = The SHAP explainer. Needed to explain predictions.

# Save metadata (text info about the model) as JSON
metadata = {
    'model_name': best_model_name,
    'threshold': float(best_thresh),
    'feature_names': list(X.columns),
    'target_names': ['Benign', 'Malignant'],
    'metrics': {
        'accuracy': float(optimal_results[best_model_name]['accuracy']),
        'precision': float(optimal_results[best_model_name]['precision']),
        'recall': float(optimal_results[best_model_name]['recall']),
        'f1': float(optimal_results[best_model_name]['f1']),
        'f2': float(optimal_results[best_model_name]['f2']),
        'roc_auc': float(roc_auc_score(y, y_final_proba))
    }
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
# indent=2 = Pretty formatting (easy to read)

print("Saved production artifacts:")
print("   • breast_cancer_model.pkl  (the trained model)")
print("   • scaler.pkl               (the feature scaler)")
print("   • shap_explainer.pkl       (the explainability engine)")
print("   • model_metadata.json      (threshold + metrics + feature names)")

# ==============================================================================
# STEP 9: CLINICAL BUSINESS REPORT
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 9: Clinical Business Report")
print("=" * 70)

# Unpack the confusion matrix
tn, fp, fn, tp = cm.ravel()
# tn = True Negative (predicted healthy, actually healthy)
# fp = False Positive (predicted cancer, actually healthy) -> False alarm
# fn = False Negative (predicted healthy, actually cancer) -> DANGEROUS
# tp = True Positive (predicted cancer, actually cancer) -> Correct

sensitivity = tp / (tp + fn) * 100    # Same as Recall
specificity = tn / (tn + fp) * 100    # Of healthy people, how many correctly cleared?
ppv = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0  # Positive Predictive Value
npv = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0  # Negative Predictive Value

report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║     BREAST CANCER SCREENING AI — CLINICAL DECISION SUPPORT         ║
╚══════════════════════════════════════════════════════════════════════╝

MODEL: {best_model_name}
OPTIMAL THRESHOLD: {best_thresh:.3f}  (NOT default 0.5 — tuned for F2-score)

WHY THIS THRESHOLD?
  • Default 0.5 treats False Positives and False Negatives equally.
  • In oncology, a False Negative (missed cancer) is 100x worse than
    a False Positive (unnecessary biopsy).
  • F2-score weights Recall 2x higher than Precision.
  • This threshold catches the MAXIMUM cancers while keeping false
    alarms clinically manageable.

PERFORMANCE METRICS:
  ┌────────────────────────────────────────────────────────────────┐
  │ Sensitivity (Cancer Detection Rate):  {sensitivity:.1f}%        │
  │   → Of 100 women WITH cancer, model flags {sensitivity:.0f}    │
  │                                                                │
  │ Specificity (Healthy Clearance):     {specificity:.1f}%        │
  │   → Of 100 healthy women, model clears {specificity:.0f}       │
  │                                                                │
  │ PPV (Flagged cases that ARE cancer): {ppv:.1f}%               │
  │ NPV (Cleared cases that ARE healthy): {npv:.1f}%              │
  │                                                                │
  │ False Negatives: {fn} out of {tp+fn} cancers MISSED           │
  │ False Positives: {fp} healthy women flagged unnecessarily     │
  └────────────────────────────────────────────────────────────────┘

BUSINESS / CLINICAL IMPACT:
  1. TRIAGE TOOL: Pre-screen mammograms before radiologist review.
     → Reduces radiologist workload by ~{specificity:.0f}%
  2. SAFETY NET: Catches {sensitivity:.0f}% of cancers that might be
     missed in early screening.
  3. EXPLAINABILITY: SHAP values tell the doctor EXACTLY which cell
     measurements (radius, texture, concavity) drove the alert.
  4. COST: A missed cancer costs $100K+ in late-stage treatment.
     A false alarm costs ~$500 for a biopsy. This model optimizes
     for the outcome that saves lives AND money.

NEXT STEP: Deploy via FastAPI (see app.py)
"""
print(report)

with open("clinical_report.txt", "w") as f:
    f.write(report)
print("[✓] Saved: clinical_report.txt")
print("\n>>> TRAINING COMPLETE. Run: python breast_cancer_v2_training.py")
