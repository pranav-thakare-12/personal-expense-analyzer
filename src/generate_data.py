"""Make a realistic SYNTHETIC UPI/bank transaction file (no real personal data)."""
from pathlib import Path
import numpy as np
import pandas as pd

# category: (merchants, lognormal mu, sigma, weight)
CATS = {
    "Food": (["SWIGGY", "ZOMATO", "DOMINOS", "MCDONALDS", "CAFE COFFEE DAY", "CHAI POINT"], 5.6, 0.5, 0.30),
    "Groceries": (["BIGBASKET", "DMART", "BLINKIT", "ZEPTO", "RELIANCE FRESH"], 6.4, 0.5, 0.18),
    "Transport": (["UBER", "OLA", "IRCTC", "HP PETROL PUMP", "RAPIDO", "METRO CARD"], 5.0, 0.6, 0.17),
    "Shopping": (["AMAZON", "FLIPKART", "MYNTRA", "AJIO", "DECATHLON"], 7.2, 0.7, 0.12),
    "Entertainment": (["NETFLIX", "SPOTIFY", "BOOKMYSHOW", "PVR CINEMAS", "STEAM GAMES"], 5.8, 0.5, 0.08),
    "Health": (["APOLLO PHARMACY", "MEDPLUS", "PRACTO", "CULT FIT"], 6.3, 0.6, 0.05),
    "Education": (["UDEMY", "COURSERA", "KINDLE", "UNACADEMY"], 6.5, 0.5, 0.03),
    "Others": (["RAHUL KUMAR", "AMIT SHARMA", "PRIYA PATIL", "SNEHA DESHMUKH", "ATM CASH"], 6.5, 0.8, 0.07),
}
SHARED = ["PAYTM", "GOOGLE PAY MERCHANT"]  # unclear merchants -> hard for the model, so accuracy stays honest
PAY_MODES = ["UPI", "UPI", "UPI", "Debit Card", "Credit Card", "Net Banking"]


def generate(path, start="2025-11-01", end="2026-10-31", seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    names = list(CATS)
    weights = np.array([CATS[c][3] for c in names])
    weights = weights / weights.sum()

    def add(date, merchant, amount, cat, star=False):
        ref = rng.integers(10**8, 10**9)
        tail = f"*ORDER {rng.integers(1000, 9999)}" if star else ""
        rows.append((date, f"UPI/{merchant}{tail}/{ref}", round(float(amount), 2), cat,
                     str(rng.choice(PAY_MODES))))

    for day in pd.date_range(start, end):
        weekend = day.dayofweek >= 5
        for _ in range(rng.poisson(3.6 if weekend else 2.8)):
            cat = str(rng.choice(names, p=weights))
            merchants, mu, sigma, _w = CATS[cat]
            merchant = rng.choice(SHARED) if rng.random() < 0.08 else rng.choice(merchants)
            amount = rng.lognormal(mu, sigma)
            if weekend and cat in ("Food", "Entertainment"):
                amount *= 1.3
            if cat == "Shopping" and day.month in (10, 11, 12):  # festival season
                amount *= 1.4
            add(day, str(merchant), max(amount, 10), cat, star=cat in ("Food", "Groceries") and merchant not in SHARED)
        # fixed monthly bills
        if day.day == 3:
            add(day, "RENT PAYMENT", 12000, "Bills")
        if day.day == 8:
            add(day, "JIO RECHARGE", 299, "Bills")
        if day.day == 12:
            add(day, "AIRTEL BROADBAND", 799, "Bills")
        if day.day == 15:
            add(day, "MSEDCL ELECTRICITY BILL", rng.normal(1400, 250), "Bills")

    # a few unusual spends to detect later
    all_days = pd.date_range(start, end)
    for _ in range(8):
        d = all_days[rng.integers(len(all_days))]
        if rng.random() < 0.6:
            add(d, "CROMA ELECTRONICS", rng.uniform(15000, 40000), "Shopping")
        else:
            add(d, "SWIGGY", rng.uniform(3500, 6000), "Food", star=True)

    df = pd.DataFrame(rows, columns=["date", "description", "amount", "category", "payment_mode"])
    df = df.sort_values("date").reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "transactions.csv"
    d = generate(out)
    print(f"Saved {len(d)} transactions to {out}")
