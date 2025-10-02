"""visualize.py
Creates:
- regression scatter with predicted vs true
- residual histogram
- regression coefficients (if multiple features)
"""

import argparse, json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def load_metrics(path):
    with open(path) as f:
        return json.load(f)

def plot_pred_vs_true(y_true, y_pred, outpath):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], linestyle='--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Predicted vs True')
    plt.savefig(outpath)
    plt.close()

def plot_residuals(y_true, y_pred, outpath):
    res = y_true - y_pred
    plt.figure()
    plt.hist(res, bins=30)
    plt.xlabel('Residual')
    plt.title('Residuals Distribution')
    plt.savefig(outpath)
    plt.close()

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model = joblib.load(args.model)
    df = pd.read_csv(args.data_path)
    X = df.drop(columns=[args.target_column], errors='ignore')
    y = df[args.target_column]

    # Preprocess similarly to training script? For quick view, drop rows with NA in y and numeric fill
    df = df.dropna(subset=[args.target_column])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if args.target_column in numeric_cols: numeric_cols.remove(args.target_column)
    X = df[numeric_cols]
    y = df[args.target_column]

    # Make predictions (note: if model expects different features, this may fail; use train.py pipeline in practice)
    preds = model.predict(X)

    plot_pred_vs_true(y, preds, os.path.join(args.output_dir, 'pred_vs_true.png'))
    plot_residuals(y, preds, os.path.join(args.output_dir, 'residuals.png'))

    # If coefficients are available:
    try:
        coef = model.coef_
        features = X.columns.tolist()
        coef_df = pd.DataFrame({'feature': features, 'coefficient': coef})
        coef_df = coef_df.reindex(coef_df.coefficient.abs().sort_values(ascending=False).index)
        coef_df.to_csv(os.path.join(args.output_dir, 'coefficients.csv'), index=False)
    except Exception as e:
        print('Could not extract coefficients:', e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, help='Path to metrics json', required=False)
    parser.add_argument('--model', type=str, default='models/linear_regression.pkl')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--target-column', type=str, default='SalePrice')
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    main(args)