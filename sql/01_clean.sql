-- 01_clean.sql : raw -> analysis-ready users
-- raw_users / raw_sessions / raw_purchases are registered by src/02_analysis.py

-- 1) exact duplicate rows removed, missing country labelled (not dropped)
CREATE OR REPLACE TABLE users_dedup AS
SELECT DISTINCT
    user_id,
    CAST(install_date AS DATE)      AS install_date,
    variant, platform, channel,
    COALESCE(country, 'Unknown')    AS country
FROM raw_users;

-- 2) bot-like accounts: average rounds per active day far above any human range
--    (threshold chosen after inspecting the distribution, see README "Data quality")
CREATE OR REPLACE TABLE bot_users AS
SELECT user_id
FROM raw_sessions
GROUP BY user_id
HAVING AVG(rounds) > 150;

CREATE OR REPLACE TABLE users_final AS
SELECT * FROM users_dedup
WHERE user_id NOT IN (SELECT user_id FROM bot_users);
