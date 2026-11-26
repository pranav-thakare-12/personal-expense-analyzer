-- All queries use :start and :end (YYYY-MM-DD). Format "-- name: <key>" is read by src/db.py

-- name: kpis
SELECT ROUND(SUM(amount), 2) AS total_spend,
       COUNT(*) AS txns,
       ROUND(AVG(amount), 2) AS avg_txn,
       COUNT(DISTINCT txn_date) AS active_days
FROM transactions
WHERE txn_date BETWEEN :start AND :end;

-- name: monthly_spend
SELECT strftime('%Y-%m', txn_date) AS month, ROUND(SUM(amount), 2) AS total
FROM transactions
WHERE txn_date BETWEEN :start AND :end
GROUP BY month
ORDER BY month;

-- name: category_share
SELECT c.category_name AS category,
       ROUND(SUM(t.amount), 2) AS total,
       ROUND(100.0 * SUM(t.amount) / SUM(SUM(t.amount)) OVER (), 1) AS pct
FROM transactions t
JOIN categories c USING (category_id)
WHERE t.txn_date BETWEEN :start AND :end
GROUP BY c.category_name
ORDER BY total DESC;

-- name: top_merchants
SELECT merchant, COUNT(*) AS txns, ROUND(SUM(amount), 2) AS total
FROM transactions
WHERE txn_date BETWEEN :start AND :end
GROUP BY merchant
ORDER BY total DESC
LIMIT 10;

-- name: mom_change
WITH m AS (
    SELECT strftime('%Y-%m', txn_date) AS month, SUM(amount) AS total
    FROM transactions
    WHERE txn_date BETWEEN :start AND :end
    GROUP BY month
)
SELECT month,
       ROUND(total, 2) AS total,
       ROUND(LAG(total) OVER (ORDER BY month), 2) AS prev_total,
       ROUND(100.0 * (total - LAG(total) OVER (ORDER BY month)) / LAG(total) OVER (ORDER BY month), 1) AS mom_pct
FROM m
ORDER BY month;

-- name: weekend_vs_weekday
SELECT CASE WHEN strftime('%w', txn_date) IN ('0', '6') THEN 'Weekend' ELSE 'Weekday' END AS day_type,
       COUNT(DISTINCT txn_date) AS days,
       ROUND(SUM(amount), 2) AS total,
       ROUND(SUM(amount) / COUNT(DISTINCT txn_date), 2) AS avg_per_day
FROM transactions
WHERE txn_date BETWEEN :start AND :end
GROUP BY day_type;

-- name: category_month
SELECT strftime('%Y-%m', t.txn_date) AS month, c.category_name AS category, ROUND(SUM(t.amount), 2) AS total
FROM transactions t
JOIN categories c USING (category_id)
WHERE t.txn_date BETWEEN :start AND :end
GROUP BY month, category;
