"""SQLite layer: builds tables from a DataFrame and runs the named queries in sql/queries.sql."""
import sqlite3
from pathlib import Path
import pandas as pd

SCHEMA = """
CREATE TABLE categories (category_id INTEGER PRIMARY KEY AUTOINCREMENT, category_name TEXT UNIQUE);
CREATE TABLE transactions (
    txn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_date TEXT NOT NULL,
    description TEXT,
    merchant TEXT,
    amount REAL NOT NULL,
    category_id INTEGER REFERENCES categories(category_id),
    payment_mode TEXT
);
"""
QUERY_FILE = Path(__file__).resolve().parent.parent / "sql" / "queries.sql"


def extract_merchant(description: str) -> str:
    """'UPI/SWIGGY*ORDER 4821/123' -> 'SWIGGY'"""
    parts = str(description).split("/")
    name = parts[1] if len(parts) > 1 else parts[0]
    return name.split("*")[0].strip().upper()


def load_queries(path=QUERY_FILE) -> dict:
    queries, name = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith("-- name:"):
            name = line.split(":", 1)[1].strip()
            queries[name] = []
        elif name and not line.startswith("--"):
            queries[name].append(line)
    return {k: "\n".join(v).strip() for k, v in queries.items()}


QUERIES = load_queries()


def build_db(df: pd.DataFrame) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    cats = sorted(df["category"].unique())
    conn.executemany("INSERT INTO categories(category_name) VALUES (?)", [(c,) for c in cats])
    cid = {c: i + 1 for i, c in enumerate(cats)}
    rows = [(r.date.strftime("%Y-%m-%d"), r.description, r.merchant, float(r.amount),
             cid[r.category], r.payment_mode) for r in df.itertuples()]
    conn.executemany(
        "INSERT INTO transactions(txn_date, description, merchant, amount, category_id, payment_mode) "
        "VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    return conn


def run(conn, name: str, start: str, end: str) -> pd.DataFrame:
    return pd.read_sql_query(QUERIES[name], conn, params={"start": start, "end": end})
