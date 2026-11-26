# 💸 Personal Expense Analyzer

An end-to-end data project: **SQL → Pandas/NumPy → Seaborn → Machine Learning → Streamlit app**.
It reads bank/UPI transactions, finds where money goes, auto-labels categories, flags unusual spends, and forecasts next month.

> The sample data is **synthetic** (made with NumPy). No real personal data is in this repo.

## Features
| Area | What it does | Tools |
|---|---|---|
| SQL | Tables + CTEs, JOINs, window functions (`LAG`, `SUM() OVER`) in `sql/queries.sql` | SQLite |
| Analytics | KPIs, month-over-month change, weekend vs weekday, top merchants, savings rate | Pandas, NumPy |
| Visuals | Heatmap, box plot, line and bar charts, confusion matrix | Seaborn |
| Category AI | Predicts category from text like `UPI/SWIGGY*ORDER 4821/...` (TF-IDF + Logistic Regression), compared with a baseline | scikit-learn |
| Anomalies | Isolation Forest + category-wise robust z-score (flag only when both agree) | scikit-learn, NumPy |
| Forecast | Next-month spend; 3-month average vs trend model, tested with walk-forward MAE | scikit-learn |
| App | Date filter, CSV upload, 5 tabs, "try it" box for the classifier | Streamlit |

## Run it
```bash
pip install -r requirements.txt
python src/generate_data.py      # optional: make new sample data
streamlit run app.py
```
To use your own data, upload a CSV with columns `date, description, amount` (a `category` column is optional; if missing, the model predicts it).

## Folder structure
```
app.py                  Streamlit dashboard
src/generate_data.py    synthetic data generator
src/db.py               SQLite tables + query runner
src/ml.py               classifier, anomaly detection, forecast
sql/queries.sql         all SQL queries
data/transactions.csv   sample data
```

## Results on sample data (your numbers may differ slightly)
- Category classifier: ~95% accuracy vs ~30% for the "always guess top class" baseline.
- Anomaly detector finds the injected big purchases (electronics, very large food orders).
- Forecast: compares two methods and picks the one with lower error.

## Honest limits
- Sample data is synthetic and cleaner than real bank data, so real-world accuracy will be lower.
- Only 12 months of data, so the forecast is simple and meant to show the method, not to be exact.

## Ideas to extend
Add a budget alert system, recurring-payment detection, PDF statement parsing, or deploy on Streamlit Community Cloud.
