# Linear Regression Project (Task 3)

This repository contains a complete deliverable for **Task 3: Linear Regression** from the provided internship PDF.

## Contents
- `src/train.py` : Main training script (loads data, preprocessing, trains Linear Regression, evaluates, saves model & metrics).
- `src/visualize.py` : Script to create diagnostic plots (regression line, residuals, feature importance for multiple regression).
- `requirements.txt` : Python dependencies.
- `report/Report.md` : Brief report summarizing approach, results, and answers to interview questions from the PDF.
- `data/` : Place your dataset here (not included).
- `models/` : Trained model will be saved here (`linear_regression.pkl`).
- `.gitignore` : Ignore large files and outputs.

## How to use
1. Download the dataset suggested in the PDF (example Kaggle link):
   `https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction`
   Put the CSV file as `data/housing.csv` (or change `--data-path` argument).

2. Create virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

3. Train the model:
```bash
python src/train.py --data-path data/housing.csv --target-column SalePrice --output-dir outputs
```

4. Visualize:
```bash
python src/visualize.py --metrics outputs/metrics.json --model models/linear_regression.pkl --data-path data/housing.csv --target-column SalePrice --output-dir outputs
```

5. find:
- `outputs/metrics.json` : MAE, MSE, RMSE, R2.
- `models/linear_regression.pkl` : Pickled trained model.
- `outputs/` : Plots and saved artifacts.

## Notes
- Dataset is **NOT** included due to licensing and size; download it from Kaggle and place in `data/`.
- The code is written to work with a typical housing CSV;  to adapt column names (target column, categorical fields).
