"""Machine learning part: category classifier, anomaly detection, spend forecast."""
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def clean_text(s: str) -> str:
    """Lowercase and remove digits/symbols so 'ORDER 4821' and 'ORDER 9999' look the same."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", str(s).lower())).strip()


# ---------- 1. Category classifier (NLP) ----------
def train_classifier(df: pd.DataFrame) -> dict:
    X, y = df["description"].map(clean_text), df["category"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    labels = sorted(y.unique())
    majority = y_tr.value_counts().idxmax()  # baseline: always guess the most common category
    return {
        "model": pipe,
        "accuracy": accuracy_score(y_te, pred),
        "baseline_accuracy": accuracy_score(y_te, [majority] * len(y_te)),
        "report": pd.DataFrame(classification_report(y_te, pred, output_dict=True, zero_division=0)).T.round(2),
        "confusion": pd.DataFrame(confusion_matrix(y_te, pred, labels=labels), index=labels, columns=labels),
    }


def predict_category(model, descriptions) -> list:
    return list(model.predict([clean_text(d) for d in descriptions]))


# ---------- 2. Anomaly detection ----------
def detect_anomalies(df: pd.DataFrame, contamination=0.01) -> pd.DataFrame:
    """Flags a transaction if Isolation Forest OR a category-wise robust z-score says it is unusual."""
    out = df.copy()
    log_amt = np.log1p(out["amount"])
    grp = log_amt.groupby(out["category"])
    med = grp.transform("median")
    mad = grp.transform(lambda s: (s - s.median()).abs().median()).replace(0, np.nan)
    out["z_score"] = (0.6745 * (log_amt - med) / mad).fillna(0).round(2)
    feats = pd.DataFrame({"log_amt": log_amt, "z": out["z_score"]})
    iso = IsolationForest(contamination=contamination, random_state=42).fit(feats)
    out["iso_flag"] = iso.predict(feats) == -1
    out["z_flag"] = out["z_score"].abs() > 4
    out["anomaly"] = out["iso_flag"] & out["z_flag"]  # both agree -> fewer false alarms
    return out


# ---------- 3. Forecast ----------
def forecast_next_month(df: pd.DataFrame) -> dict:
    monthly = df.set_index("date")["amount"].resample("MS").sum()
    y = monthly.values
    n = len(y)
    if n < 6:
        return {"error": "Need at least 6 months of data to forecast."}

    def baseline(hist):  # average of last 3 months
        return hist[-3:].mean()

    def trend(hist):  # straight-line trend
        t = np.arange(len(hist)).reshape(-1, 1)
        return LinearRegression().fit(t, hist).predict([[len(hist)]])[0]

    # walk-forward test on the last 3 months
    errs = {"baseline": [], "trend": []}
    for i in range(n - 3, n):
        errs["baseline"].append(abs(baseline(y[:i]) - y[i]))
        errs["trend"].append(abs(trend(y[:i]) - y[i]))
    mae = {k: float(np.mean(v)) for k, v in errs.items()}
    best = min(mae, key=mae.get)
    pred = baseline(y) if best == "baseline" else trend(y)

    cat_monthly = df.groupby([pd.Grouper(key="date", freq="MS"), "category"])["amount"].sum().unstack(fill_value=0)
    cat_forecast = cat_monthly.tail(3).mean().round(0).sort_values(ascending=False)
    return {"monthly": monthly, "next_month": monthly.index[-1] + pd.offsets.MonthBegin(1),
            "prediction": float(pred), "mae": mae, "best": best, "category_forecast": cat_forecast}
