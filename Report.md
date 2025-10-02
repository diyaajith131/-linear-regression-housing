# Report — Linear Regression (Task 3)

## Objective
Implement simple & multiple linear regression, evaluate performance, and interpret results.

## Dataset
Use the Kaggle Housing Price Prediction dataset (place `housing.csv` in `data/`).

## Approach
1. **Load & preprocess**: drop missing target rows, simple imputation (median for numeric, mode for categorical), one-hot encode categoricals.
2. **Split**: train/test split (80/20).
3. **Model**: `sklearn.linear_model.LinearRegression`.
4. **Evaluation**: MAE, MSE, RMSE, R².
5. **Save**: model pickled to `models/linear_regression.pkl`, metrics in `outputs/metrics.json`.

## Results (example)
*(After running `train.py` you will find exact numbers in `outputs/metrics.json`.)*

## Interview Questions (short answers)
1. **Assumptions of linear regression**: linearity, independence of errors, homoscedasticity (constant variance), normality of errors (for inference), no perfect multicollinearity.
2. **Interpreting coefficients**: In a multiple linear regression, a coefficient is the expected change in the target for a one-unit increase in the feature, holding other features constant.
3. **R² score**: Fraction of variance explained by the model (0 to 1). Higher is better but can be misleading if overfitting.
4. **MSE vs MAE**: MSE penalizes larger errors more strongly (squared), useful when large errors are particularly undesirable. MAE is more robust to outliers.
5. **Detect multicollinearity**: Use Variance Inflation Factor (VIF), condition number, or examine correlation matrix.
6. **Simple vs multiple regression**: Simple uses one predictor; multiple uses two or more predictors.
7. **Linear regression for classification**: No — it's for continuous targets. You could threshold predictions but proper classification models are preferred.
8. **Violating assumptions**: Leads to biased/inefficient estimates, invalid inference, poor predictive performance.

## How to improve
- Better preprocessing (treat outliers, transform skewed features).
- Feature engineering (interaction terms, polynomial features).
- Regularization (Ridge/Lasso) for multicollinearity.
- Cross-validation for robust performance estimation.