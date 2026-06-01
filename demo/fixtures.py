"""Synthetic but realistically messy fixtures.

Two databases, multiple schemas, mixed dialects, deeply nested columns
(Snowflake VARIANT/OBJECT, BigQuery ARRAY<STRUCT<...>>), and a set of
spaghetti queries exhibiting common anti-patterns.
"""

# ---- Database 1: PROD (Snowflake-style) --------------------------------------
SNOWFLAKE_DDL = """
CREATE OR REPLACE TABLE PROD.CORE.USERS (
    user_id        NUMBER PRIMARY KEY,
    email          VARCHAR NOT NULL,
    full_name      VARCHAR,
    country        VARCHAR,
    signup_at      TIMESTAMP_NTZ,
    plan_tier      VARCHAR,
    attributes     VARIANT,
    billing_address OBJECT(street STRING, city STRING, zip STRING)
);

CREATE OR REPLACE TABLE PROD.CORE.ACCOUNTS (
    account_id     NUMBER PRIMARY KEY,
    owner_user_id  NUMBER,
    mrr            NUMBER,
    created_at     TIMESTAMP_NTZ,
    is_active      BOOLEAN,
    FOREIGN KEY (owner_user_id) REFERENCES PROD.CORE.USERS(user_id)
);

CREATE OR REPLACE TABLE PROD.CORE.SUBSCRIPTIONS (
    subscription_id NUMBER PRIMARY KEY,
    account_id      NUMBER,
    user_id         NUMBER,
    started_at      TIMESTAMP_NTZ,
    ended_at        TIMESTAMP_NTZ,
    monthly_amount  NUMBER,
    status          VARCHAR
);

CREATE OR REPLACE TABLE PROD.BILLING.INVOICES (
    invoice_id     NUMBER PRIMARY KEY,
    account_id     NUMBER,
    amount         NUMBER,
    paid_at        TIMESTAMP_NTZ,
    line_items     ARRAY
);
"""

# ---- Database 2: ANALYTICS (BigQuery-style, heavily nested) ------------------
BIGQUERY_DDL = """
CREATE TABLE ANALYTICS.EVENTS.PRODUCT_EVENTS (
    event_id     STRING,
    user_id      INT64,
    session_id   STRING,
    event_time   TIMESTAMP,
    event_name   STRING,
    device       STRUCT<os STRING, browser STRING, ip_address STRING>,
    properties   ARRAY<STRUCT<key STRING, value STRING>>,
    page         STRUCT<url STRING, referrer STRING, utm STRUCT<source STRING, medium STRING, campaign STRING>>
);

CREATE TABLE ANALYTICS.EVENTS.SUPPORT_TICKETS (
    ticket_id    INT64,
    user_id      INT64,
    created_at   TIMESTAMP,
    channel      STRING,
    body         STRING,
    resolution_minutes INT64,
    tags         ARRAY<STRING>
);

CREATE TABLE ANALYTICS.MART.USER_FEATURES (
    user_id          INT64,
    feature_date     DATE,
    sessions_7d      INT64,
    revenue_30d      FLOAT64,
    ltv_pred         FLOAT64,
    churn_score      FLOAT64
);
"""

# ---- Spaghetti SQL -----------------------------------------------------------
SPAGHETTI_QUERIES = {
    # 1. deep nesting, no CTEs, SELECT *, multiple DISTINCTs (fan-out band-aid)
    "retention_blob": """
        SELECT DISTINCT u.user_id, u.country,
               (SELECT COUNT(DISTINCT e.session_id)
                  FROM ANALYTICS.EVENTS.PRODUCT_EVENTS e
                 WHERE e.user_id = u.user_id
                   AND e.event_time > (SELECT MIN(s.started_at)
                                         FROM PROD.CORE.SUBSCRIPTIONS s
                                        WHERE s.user_id = u.user_id
                                          AND s.account_id IN (SELECT account_id
                                                                 FROM PROD.CORE.ACCOUNTS
                                                                WHERE is_active))) AS sessions
          FROM PROD.CORE.USERS u, PROD.CORE.ACCOUNTS a
         WHERE u.user_id = a.owner_user_id
    """,

    # 2. many joins, fan-out across one-to-many without pre-aggregation
    "revenue_rollup": """
        WITH base AS (
            SELECT a.account_id, a.mrr, s.monthly_amount, i.amount
              FROM PROD.CORE.ACCOUNTS a
              JOIN PROD.CORE.SUBSCRIPTIONS s ON a.account_id = s.account_id
              JOIN PROD.BILLING.INVOICES i ON i.account_id = a.account_id
              JOIN PROD.CORE.USERS u ON u.user_id = a.owner_user_id
              JOIN ANALYTICS.MART.USER_FEATURES f ON f.user_id = u.user_id
              JOIN ANALYTICS.EVENTS.SUPPORT_TICKETS t ON t.user_id = u.user_id
              JOIN ANALYTICS.EVENTS.PRODUCT_EVENTS e ON e.user_id = u.user_id
        )
        SELECT account_id, SUM(amount) FROM base GROUP BY account_id
    """,

    # 3. explicit cross join + correlated subquery
    "cohort_cross": """
        SELECT c.cohort_month, u.user_id
          FROM PROD.CORE.USERS u
          CROSS JOIN (SELECT DISTINCT DATE_TRUNC('month', signup_at) AS cohort_month
                        FROM PROD.CORE.USERS) c
         WHERE EXISTS (SELECT 1 FROM PROD.CORE.SUBSCRIPTIONS s
                        WHERE s.user_id = u.user_id AND s.status = 'active')
    """,

    # 4. clean reference query (should score low)
    "clean_metric": """
        SELECT u.country, COUNT(*) AS users
          FROM PROD.CORE.USERS u
          JOIN PROD.CORE.ACCOUNTS a ON a.owner_user_id = u.user_id
         WHERE a.is_active
         GROUP BY u.country
    """,
}
