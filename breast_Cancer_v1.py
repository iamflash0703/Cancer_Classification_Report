"""
Project 3: Basic Classification Model

Goal: Predict whether a breast tumor is malignant (cancerous) or benign
(non-cancerous) based on cell measurements, using classification models.
"""

# ---------- STEP 0: Import Libraries ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer          # built-in dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler          # feature scaling
from sklearn.linear_model import LogisticRegression       # model 1
from sklearn.tree import DecisionTreeClassifier            # model 2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

sns.set_style("darkgrid")

# ---------- STEP 1: Data Preparation ----------
# Scikit-learn ships this dataset directly - no download needed.
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

print("Dataset shape:", X.shape)
print("Target classes:", dict(zip(data.target_names, [0, 1])))
print("\nClass distribution:\n", y.value_counts())
print("\nFirst 5 rows of features:\n", X.head())

# ---------- STEP 2: Data Splitting ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    # stratify=y keeps the same class ratio in both train and test sets
)
print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ---------- STEP 2b: Feature Scaling ----------
# Logistic Regression is sensitive to feature scale (some features are
# measured in very different ranges), so we standardize them
# (mean=0, std=1) using the training data statistics only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- STEP 3: Model Building & Training ----------
# Model 1: Logistic Regression (simple, interpretable, good baseline)
log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train_scaled, y_train)

# Model 2: Decision Tree (non-linear, easy to visualize decision rules)
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)  # trees don't need feature scaling

# ---------- STEP 4: Predictions ----------
log_pred = log_model.predict(X_test_scaled)
tree_pred = tree_model.predict(X_test)

# ---------- STEP 5: Model Evaluation ----------
def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    return acc, prec, rec, f1

log_metrics = evaluate("Logistic Regression", y_test, log_pred)
tree_metrics = evaluate("Decision Tree", y_test, tree_pred)

print("\nDetailed Classification Report (Logistic Regression):\n",
      classification_report(y_test, log_pred, target_names=data.target_names))

# ---------- STEP 6: Visualization ----------

# 6a. Confusion Matrix for Logistic Regression
cm = confusion_matrix(y_test, log_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="mako",
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Confusion Matrix — Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# 6b. Model comparison bar chart
metrics_df = pd.DataFrame({
    "Logistic Regression": log_metrics,
    "Decision Tree": tree_metrics,
}, index=["Accuracy", "Precision", "Recall", "F1-Score"])

plt.figure(figsize=(8, 5))
metrics_df.plot(kind="bar", ax=plt.gca(), color=["#4C9AFF", "#2ECC71"])
plt.title("Model Comparison — Logistic Regression vs Decision Tree")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# 6c. Feature importance from Decision Tree (top 10)
importances = pd.Series(tree_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top 10 Important Features (Decision Tree)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("\nAll plots saved successfully.")

# ---------- STEP 7: Key Insights ----------
print("\nKey Insights:")
print(f"1. Logistic Regression achieved {log_metrics[0]*100:.1f}% accuracy, "
      f"Decision Tree achieved {tree_metrics[0]*100:.1f}% accuracy.")
print(f"2. Most important feature for prediction: '{top_features.index[0]}'")
print("3. High recall matters most in medical diagnosis — missing a "
      "malignant case (false negative) is far more dangerous than a "
      "false alarm (false positive).")
