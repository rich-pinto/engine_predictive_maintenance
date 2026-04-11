
import joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

test_df = pd.read_csv(DATA_DIR / "test.csv")
X_test = test_df.drop(columns=["Engine Condition"])
y_test = test_df["Engine Condition"]

model = joblib.load(MODELS_DIR / "final_trained_pipeline.joblib")

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "Sensitivity", "Specificity", "F1", "ROC_AUC"],
    "Value": [
        accuracy_score(y_test, preds),
        precision_score(y_test, preds, zero_division=0),
        recall_score(y_test, preds, zero_division=0),
        sensitivity,
        specificity,
        f1_score(y_test, preds, zero_division=0),
        roc_auc_score(y_test, probs) if probs is not None else None
    ]
})

pred_output = test_df.copy()
pred_output["Predicted_Engine_Condition"] = preds
if probs is not None:
    pred_output["Predicted_Probability"] = probs

results.to_csv(ARTIFACTS_DIR / "final_pipeline_metrics.csv", index=False)
pred_output.to_csv(ARTIFACTS_DIR / "final_pipeline_predictions.csv", index=False)

print("Inference pipeline executed successfully.")
print(results)
print("Saved metrics to:", ARTIFACTS_DIR / "final_pipeline_metrics.csv")
print("Saved predictions to:", ARTIFACTS_DIR / "final_pipeline_predictions.csv")
