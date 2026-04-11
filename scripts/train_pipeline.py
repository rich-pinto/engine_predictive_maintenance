
import json
import joblib
from pathlib import Path
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# Resolve the project root based on the script location:
# .../engine_predictive_maintenance/scripts/train_pipeline.py
# -> project root is parent.parent
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train.csv")
X_train = train_df.drop(columns=["Engine Condition"])
y_train = train_df["Engine Condition"]

with open(MODELS_DIR / "final_model_config.json", "r", encoding="utf-8") as f:
    final_model_config = json.load(f)

selected_model_name = final_model_config["selected_model_name"]

# Define reusable candidate pipelines matching the notebook final stage.
base_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(random_state=42, max_iter=2000, class_weight="balanced"))
])

base_rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_estimators=300,
    max_depth=8,
    min_samples_split=10
)

base_gb = GradientBoostingClassifier(
    random_state=42,
    learning_rate=0.05,
    max_depth=2,
    n_estimators=100
)

estimators = [("lr", base_lr), ("rf", base_rf), ("gb", base_gb)]

if XGB_AVAILABLE:
    base_xgb = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3
    )
    estimators.append(("xgb", base_xgb))

candidate_models = {
    "Interim_GradientBoosting": base_gb,
    "Refined_GradientBoosting": base_gb,
    "SoftVotingEnsemble": VotingClassifier(estimators=estimators, voting="soft"),
    "StackingEnsemble": StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        passthrough=False,
        n_jobs=-1
    )
}

final_model = candidate_models[selected_model_name]
final_model.fit(X_train, y_train)

output_path = MODELS_DIR / "final_trained_pipeline.joblib"
joblib.dump(final_model, output_path)

print("Training pipeline executed successfully.")
print("Saved trained pipeline to:", output_path)
