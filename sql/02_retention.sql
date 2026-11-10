-- 02_retention.sql : activity by day-since-install, per-user retention flags

CREATE OR REPLACE TABLE activity AS
SELECT
    s.user_id,
    DATE_DIFF('day', u.install_date, CAST(s.session_date AS DATE)) AS day_n,
    s.sessions, s.rounds, s.minutes
FROM raw_sessions s
JOIN users_final u USING (user_id);

-- one row per user: classic "active on exactly day N" retention + early engagement
CREATE OR REPLACE TABLE user_metrics AS
SELECT
    u.user_id, u.variant, u.platform, u.channel, u.country,
    MAX(CASE WHEN a.day_n = 1  THEN 1 ELSE 0 END) AS d1,
    MAX(CASE WHEN a.day_n = 7  THEN 1 ELSE 0 END) AS d7,
    MAX(CASE WHEN a.day_n = 14 THEN 1 ELSE 0 END) AS d14,
    COALESCE(SUM(CASE WHEN a.day_n BETWEEN 0 AND 6 THEN a.rounds END), 0)                 AS rounds_7d,
    COUNT(DISTINCT CASE WHEN a.day_n BETWEEN 0 AND 6 THEN a.day_n END)                    AS active_days_7d
FROM users_final u
LEFT JOIN activity a USING (user_id)
GROUP BY u.user_id, u.variant, u.platform, u.channel, u.country;

-- retention curve (day 0..14) by variant
CREATE OR REPLACE TABLE retention_curve AS
WITH cohort AS (
    SELECT variant, COUNT(*) AS n FROM users_final GROUP BY variant
)
SELECT
    a.day_n,
    u.variant,
    COUNT(DISTINCT a.user_id)              AS retained,
    c.n                                    AS cohort_size,
    COUNT(DISTINCT a.user_id) * 1.0 / c.n  AS retention
FROM activity a
JOIN users_final u USING (user_id)
JOIN cohort c ON c.variant = u.variant
WHERE a.day_n BETWEEN 0 AND 14
GROUP BY a.day_n, u.variant, c.n
ORDER BY a.day_n, u.variant;
