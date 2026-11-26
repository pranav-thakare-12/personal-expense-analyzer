import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src import db, ml
from src.generate_data import generate

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Personal Expense Analyzer", page_icon="💸", layout="wide")
sns.set_theme(style="whitegrid")
DATA = Path(__file__).parent / "data" / "transactions.csv"


@st.cache_data
def load_sample() -> pd.DataFrame:
    if not DATA.exists():
        generate(DATA)
    df = pd.read_csv(DATA, parse_dates=["date"])
    df["merchant"] = df["description"].map(db.extract_merchant)
    return df


@st.cache_resource
def get_classifier():
    return ml.train_classifier(load_sample())


def prepare_upload(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"date", "description", "amount"} - set(df.columns)
    if missing:
        st.error(f"Your CSV must have columns: date, description, amount. Missing: {', '.join(missing)}")
        st.stop()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"]).query("amount > 0").copy()
    df["merchant"] = df["description"].map(db.extract_merchant)
    if "category" not in df.columns:
        df["category"] = ml.predict_category(get_classifier()["model"], df["description"])
    if "payment_mode" not in df.columns:
        df["payment_mode"] = "UPI"
    return df.reset_index(drop=True)


def fmt(x):
    return f"₹{x:,.0f}"


# ------------------ Sidebar ------------------
st.title("💸 Personal Expense Analyzer")
st.caption("SQL + Pandas + NumPy + Seaborn + Machine Learning, in one dashboard. Sample data is synthetic.")

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload your CSV (date, description, amount)", type="csv")
    st.caption("Your file stays in this session only. Category is predicted by the ML model if missing.")
    df_all = prepare_upload(up) if up else load_sample()
    lo, hi = df_all["date"].min().date(), df_all["date"].max().date()
    rng = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    start, end = (rng[0], rng[1]) if isinstance(rng, tuple) and len(rng) == 2 else (lo, hi)
    income = st.number_input("Monthly income (₹)", min_value=0, value=120000, step=5000)

mask = (df_all["date"].dt.date >= start) & (df_all["date"].dt.date <= end)
df = df_all[mask].reset_index(drop=True)
if df.empty:
    st.warning("No transactions in this range.")
    st.stop()
conn = db.build_db(df_all)
s, e = str(start), str(end)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔎 Insights", "🤖 Category AI", "🚨 Anomalies", "📈 Forecast"])

# ------------------ Overview ------------------
with tab1:
    k = db.run(conn, "kpis", s, e).iloc[0]
    months = max(df["date"].dt.to_period("M").nunique(), 1)
    avg_month = k.total_spend / months
    savings_rate = (1 - avg_month / income) * 100 if income else np.nan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total spend", fmt(k.total_spend))
    c2.metric("Avg per month", fmt(avg_month))
    c3.metric("Transactions", f"{int(k.txns):,}")
    c4.metric("Savings rate", f"{savings_rate:.0f}%" if income else "–")

    left, right = st.columns(2)
    monthly = db.run(conn, "monthly_spend", s, e)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.lineplot(data=monthly, x="month", y="total", marker="o", ax=ax)
    ax.set_title("Monthly spend"); ax.set_xlabel(""); ax.set_ylabel("₹")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    left.pyplot(fig)

    share = db.run(conn, "category_share", s, e)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.barplot(data=share, y="category", x="total", ax=ax)
    ax.set_title("Spend by category"); ax.set_ylabel(""); ax.set_xlabel("₹")
    right.pyplot(fig)

    st.subheader("Auto insights")
    top3 = share.head(3)
    st.write(f"• **{', '.join(top3.category)}** make up **{top3.pct.sum():.0f}%** of your spending.")
    wk = db.run(conn, "weekend_vs_weekday", s, e).set_index("day_type")
    if {"Weekend", "Weekday"} <= set(wk.index):
        diff = (wk.loc["Weekend", "avg_per_day"] / wk.loc["Weekday", "avg_per_day"] - 1) * 100
        st.write(f"• You spend **{abs(diff):.0f}% {'more' if diff > 0 else 'less'}** per day on weekends than on weekdays.")
    mom = db.run(conn, "mom_change", s, e).dropna()
    if not mom.empty:
        worst = mom.loc[mom.mom_pct.idxmax()]
        st.write(f"• Biggest jump: **{worst.month}**, up **{worst.mom_pct:.0f}%** from the month before.")

# ------------------ Insights ------------------
with tab2:
    cm = db.run(conn, "category_month", s, e).pivot(index="category", columns="month", values="total").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(cm, cmap="YlOrRd", ax=ax, cbar_kws={"label": "₹"})
    ax.set_title("Category vs month heatmap"); ax.set_xlabel(""); ax.set_ylabel("")
    st.pyplot(fig)

    a, b = st.columns(2)
    d2 = df.copy()
    d2["weekday"] = d2["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.boxplot(data=d2[d2.amount < d2.amount.quantile(0.98)], x="weekday", y="amount", order=order, ax=ax)
    ax.set_title("Transaction size by weekday"); ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    a.pyplot(fig)
    b.subheader("Top merchants")
    b.dataframe(db.run(conn, "top_merchants", s, e), hide_index=True)
    st.subheader("Month-over-month change (SQL window function)")
    st.dataframe(db.run(conn, "mom_change", s, e), hide_index=True)
    with st.expander("See the SQL used"):
        st.code(db.QUERIES["mom_change"], language="sql")

# ------------------ Category AI ------------------
with tab3:
    clf = get_classifier()
    c1, c2, c3 = st.columns(3)
    c1.metric("Model accuracy", f"{clf['accuracy']*100:.1f}%")
    c2.metric("Baseline (always guess top class)", f"{clf['baseline_accuracy']*100:.1f}%")
    c3.metric("Improvement", f"+{(clf['accuracy']-clf['baseline_accuracy'])*100:.1f} pts")
    st.caption("Model: TF-IDF (word 1-2 grams) + Logistic Regression, tested on 25% held-out data.")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(clf["confusion"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    st.pyplot(fig)
    st.dataframe(clf["report"])
    st.subheader("Try it")
    txt = st.text_input("Type a transaction description", "UPI/ZOMATO*ORDER 5521/883920111")
    if txt:
        proba = clf["model"].predict_proba([ml.clean_text(txt)])[0]
        top = np.argsort(proba)[::-1][:3]
        for i in top:
            st.write(f"**{clf['model'].classes_[i]}**: {proba[i]*100:.0f}%")

# ------------------ Anomalies ------------------
with tab4:
    an = ml.detect_anomalies(df)
    flagged = an[an.anomaly].sort_values("amount", ascending=False)
    st.write(f"Found **{len(flagged)}** unusual transactions out of {len(an):,}. "
             "A transaction is flagged only when **both** Isolation Forest and a category-wise robust z-score agree.")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.scatterplot(data=an, x="date", y="amount", hue="anomaly", palette={False: "#9bb", True: "#d33"},
                    size="anomaly", sizes={False: 15, True: 80}, ax=ax)
    ax.set_yscale("log"); ax.set_title("Transactions (red = unusual)"); ax.set_xlabel("")
    st.pyplot(fig)
    st.dataframe(flagged[["date", "merchant", "category", "amount", "z_score"]], hide_index=True)

# ------------------ Forecast ------------------
with tab5:
    fc = ml.forecast_next_month(df)
    if "error" in fc:
        st.info(fc["error"])
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Predicted spend for {fc['next_month']:%b %Y}", fmt(fc["prediction"]))
        c2.metric("Error: 3-month average", fmt(fc["mae"]["baseline"]))
        c3.metric("Error: trend model", fmt(fc["mae"]["trend"]))
        st.caption(f"Walk-forward test on the last 3 months (MAE). Best method here: **{fc['best']}**. "
                   "We always compare against a simple baseline.")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        m = fc["monthly"]
        ax.plot(m.index, m.values, marker="o", label="Actual")
        ax.plot([m.index[-1], fc["next_month"]], [m.values[-1], fc["prediction"]], "r--o", label="Forecast")
        ax.legend(); ax.set_ylabel("₹")
        st.pyplot(fig)
        st.subheader("Expected spend by category next month")
        st.bar_chart(fc["category_forecast"])
