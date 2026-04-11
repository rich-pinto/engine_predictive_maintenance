
# Engine Predictive Maintenance Model

## Final Selected Model
GradientBoosting

## Purpose
This model predicts engine condition using structured sensor readings.

## Input Features
- Engine rpm
- Lub oil pressure
- Fuel pressure
- Coolant pressure
- lub oil temp
- Coolant temp

## Output
- Engine Condition
  - 0 = normal
  - 1 = maintenance required

## Final Selection Metrics
- Sensitivity: 0.869265
- Specificity: 0.315789
- F1 Score: 0.765737
- ROC-AUC: 0.700411
- Balanced Selection Score: 0.693305

## Best Tuned Hyperparameters
{
    "learning_rate": 0.05,
    "max_depth": 2,
    "n_estimators": 100
}

## Notes
This model was selected using a risk-aware evaluation framework that considered:
- sensitivity
- specificity
- F1 score
- ROC-AUC
- generalization behavior

The model is intended as an early-warning decision-support tool for predictive maintenance.
