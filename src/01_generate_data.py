"""
Synthetic data generator: mobile puzzle game A/B test.

NOTE: This dataset is SYNTHETIC. It is generated from an explicit data-generating
process (below) so the analysis can be reproduced end-to-end. The effects are
planted by construction; the point of the project is the analysis workflow
(data quality -> experiment validity -> metrics -> segments -> decision),
not the discovery of real-world facts.

Experiment: Variant B = "Welcome Reward Pack" (free boosters during first 7 days)
            Variant A = control (no reward pack)
Planted behaviour:
  - B lifts early retention, most strongly for low-intent paid traffic (ad_network, paid_social)
  - B slightly lowers in-app-purchase conversion (free boosters substitute for purchases)
  - Realistic mess: duplicate user rows, missing country, bot-like accounts
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED, N = 42, 60_000
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

# ---------- users ----------
channels = ["organic", "paid_social", "ad_network", "referral"]
ch_share = [0.35, 0.30, 0.25, 0.10]
ch_engagement = {"organic": 1.10, "paid_social": 0.90, "ad_network": 0.80, "referral": 1.20}
ch_lift = {"organic": 0.04, "paid_social": 0.12, "ad_network": 0.20, "referral": 0.02}  # relative lift on daily activity prob

users = pd.DataFrame({
    "user_id": [f"U{i:06d}" for i in range(1, N + 1)],
    "install_date": pd.Timestamp("2026-09-01") + pd.to_timedelta(rng.integers(0, 28, N), unit="D"),
    "variant": rng.choice(["A", "B"], N),
    "platform": rng.choice(["android", "ios"], N, p=[0.62, 0.38]),
    "channel": rng.choice(channels, N, p=ch_share),
    "country": rng.choice(["KR", "US", "JP", "BR", "ID"], N, p=[0.30, 0.25, 0.20, 0.13, 0.12]),
})
treat = (users["variant"] == "B").to_numpy()

# latent engagement (user-level, independent of variant)
eng = rng.lognormal(mean=-0.1, sigma=0.5, size=N)
eng *= users["channel"].map(ch_engagement).to_numpy()
eng *= np.where(users["platform"] == "ios", 1.10, 1.0)
lift = users["channel"].map(ch_lift).to_numpy()

# ---------- daily activity (day 0..14) ----------
days = np.arange(1, 15)
base = 0.42 * days ** -0.45                      # D1 ~42%, D7 ~17.5%, D14 ~13%
fade = np.where(days <= 7, 1.0, 0.6)             # treatment effect fades after the reward window
p = base[None, :] * eng[:, None] * (1 + (treat * lift)[:, None] * fade[None, :])
active = rng.random((N, 14)) < np.clip(p, 0, 0.95)

# bot-like accounts (~0.6%), randomly spread across variants
bots = rng.random(N) < 0.006
active[bots, :] = True

mask = np.column_stack([np.ones(N, bool), active])  # day 0 always active (install day)
ui, di = np.nonzero(mask)
n_sess = 1 + rng.poisson(0.8 * eng[ui])
rounds = n_sess + rng.poisson(4.0 * n_sess)
minutes = np.round(rounds * rng.uniform(0.6, 1.4, len(ui)), 1)
is_bot_row = bots[ui]
rounds = np.where(is_bot_row, rng.integers(300, 900, len(ui)), rounds)
minutes = np.where(is_bot_row, np.round(rng.uniform(5, 40, len(ui)), 1), minutes)

sessions = pd.DataFrame({
    "user_id": users["user_id"].to_numpy()[ui],
    "session_date": users["install_date"].to_numpy()[ui] + pd.to_timedelta(di, unit="D"),
    "sessions": n_sess, "rounds": rounds, "minutes": minutes,
})

# ---------- purchases (first 14 days) ----------
p_pay = np.clip(0.045 * eng * np.where(treat, 0.88, 1.0), 0, 0.5)   # B: -12% relative payer conversion
payer = (rng.random(N) < p_pay) & ~bots
pay_idx = np.nonzero(payer)[0]
n_pur = 1 + rng.poisson(0.6, len(pay_idx))
rows_u = np.repeat(pay_idx, n_pur)
pur_day = rng.choice(np.arange(14), len(rows_u), p=np.r_[0.25, 0.15, 0.10, np.full(11, 0.50 / 11)])
amount = rng.choice([0.99, 1.99, 4.99, 9.99, 19.99, 49.99], len(rows_u), p=[0.30, 0.25, 0.22, 0.15, 0.06, 0.02])
purchases = pd.DataFrame({
    "user_id": users["user_id"].to_numpy()[rows_u],
    "purchase_date": users["install_date"].to_numpy()[rows_u] + pd.to_timedelta(pur_day, unit="D"),
    "amount_usd": amount,
})

# ---------- inject data-quality issues ----------
dups = users.sample(frac=0.008, random_state=SEED)
users = pd.concat([users, dups], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
users.loc[rng.random(len(users)) < 0.015, "country"] = np.nan

users.to_csv(OUT / "users.csv", index=False)
sessions.to_csv(OUT / "sessions.csv", index=False)
purchases.to_csv(OUT / "purchases.csv", index=False)
print(f"users={len(users):,} sessions={len(sessions):,} purchases={len(purchases):,}")
