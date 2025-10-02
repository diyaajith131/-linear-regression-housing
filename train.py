"""train.py
Simple, clear training script for linear regression (simple & multiple).
Saves model (joblib) and metrics (json) and a sample predictions CSV.
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df, target_column):
    # Basic preprocessing:
    # - drop rows with missing target
    # - simple imputation: numeric -> median, categorical -> mode
    df = df.copy()
    df = df.dropna(subset=[target_column])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Drop target from feature lists
    if target_column in numeric_cols: numeric_cols.remove(target_column)
    if target_column in cat_cols: cat_cols.remove(target_column)
    # Fill numeric
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())
    # Fill categorical and one-hot encode
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    df = load_data(args.data_path)
    X, y = preprocess(df, args.target_column)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        'MAE': float(mean_absolute_error(y_test, preds)),
        'MSE': float(mean_squared_error(y_test, preds)),
        'RMSE': float(mean_squared_error(y_test, preds, squared=False)),
        'R2': float(r2_score(y_test, preds)),
        'n_features': X.shape[1]
    }

    # Save model and metrics
    joblib.dump(model, os.path.join('models', 'linear_regression.pkl'))
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save sample predictions
    sample_out = pd.DataFrame({'y_true': y_test, 'y_pred': preds})
    sample_out.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)

    print('Training finished. Metrics:\n', json.dumps(metrics, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--target-column', type=str, default='SalePrice', help='Name of target column')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split fraction')
    args = parser.parse_args()
    train(args)