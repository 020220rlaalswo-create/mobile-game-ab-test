-- 03_monetization.sql : 14-day revenue per user (guardrail metrics)

CREATE OR REPLACE TABLE revenue_14d AS
SELECT
    p.user_id,
    SUM(p.amount_usd) AS revenue_14d,
    COUNT(*)          AS n_purchases
FROM raw_purchases p
JOIN users_final u USING (user_id)
WHERE DATE_DIFF('day', u.install_date, CAST(p.purchase_date AS DATE)) BETWEEN 0 AND 13
GROUP BY p.user_id;

CREATE OR REPLACE TABLE user_metrics_full AS
SELECT
    m.*,
    COALESCE(r.revenue_14d, 0)                       AS revenue_14d,
    COALESCE(r.n_purchases, 0)                       AS n_purchases,
    CASE WHEN r.user_id IS NOT NULL THEN 1 ELSE 0 END AS payer
FROM user_metrics m
LEFT JOIN revenue_14d r USING (user_id);
