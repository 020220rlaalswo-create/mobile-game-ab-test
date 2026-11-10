"""
Mobile game A/B test analysis  (SQL via DuckDB + statistics in Python)

Pipeline
  1. Load raw CSVs -> DuckDB, run sql/01..03 (cleaning, retention, monetization)
  2. Data quality summary + sample-ratio-mismatch (SRM) check
  3. Primary metric: D1 / D7 / D14 retention  (two-proportion z-test, CI)
  4. Engagement + guardrails: rounds, payer rate, ARPU (bootstrap), ARPPU
  5. Heterogeneity: by channel / platform, BH multiple-testing correction,
     logistic regression + interaction likelihood-ratio test
  6. Power / MDE, break-even value of an extra retained user
  7. Figures + results tables
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import duckdb
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, SQL, FIG, RES = ROOT / "data", ROOT / "sql", ROOT / "figures", ROOT / "results"
for d in (FIG, RES):
    d.mkdir(exist_ok=True)

rng = np.random.default_rng(7)
Z = stats.norm.ppf(0.975)
COL_A, COL_B, COL_GREY = "#7f8ea3", "#e4572e", "#444444"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "axes.titleweight": "bold",
})

# ----------------------------------------------------------------------------
# 1. Load + SQL
# ----------------------------------------------------------------------------
raw_users = pd.read_csv(DATA / "users.csv")
raw_sessions = pd.read_csv(DATA / "sessions.csv")
raw_purchases = pd.read_csv(DATA / "purchases.csv")

con = duckdb.connect()
con.register("raw_users", raw_users)
con.register("raw_sessions", raw_sessions)
con.register("raw_purchases", raw_purchases)
for f in sorted(SQL.glob("*.sql")):
    con.execute(f.read_text())

um = con.execute("SELECT * FROM user_metrics_full").df()
curve = con.execute("SELECT * FROM retention_curve").df()
curve.to_csv(RES / "retention_curve.csv", index=False)

# ----------------------------------------------------------------------------
# 2. Data quality + SRM
# ----------------------------------------------------------------------------
avg_rounds = raw_sessions.groupby("user_id")["rounds"].mean()
bot_ids = set(con.execute("SELECT user_id FROM bot_users").df()["user_id"])
human_avg = avg_rounds[~avg_rounds.index.isin(bot_ids)]
bot_by_variant = (raw_users.drop_duplicates("user_id").assign(bot=lambda d: d.user_id.isin(bot_ids))
                  .groupby("variant")["bot"].sum())

dq = {
    "raw_user_rows": int(len(raw_users)),
    "duplicate_user_rows": int(raw_users.duplicated().sum()),
    "missing_country_rows": int(raw_users["country"].isna().sum()),
    "bot_users_removed": int(len(bot_ids)),
    "bot_users_A": int(bot_by_variant.get("A", 0)),
    "bot_users_B": int(bot_by_variant.get("B", 0)),
    "human_avg_rounds_p99.9": float(human_avg.quantile(0.999)),
    "human_avg_rounds_max": float(human_avg.max()),
    "final_users": int(len(um)),
}

cnt = um["variant"].value_counts().sort_index()
srm_chi2, srm_p = stats.chisquare(cnt.values)
srm = {"n_A": int(cnt["A"]), "n_B": int(cnt["B"]), "chi2": float(srm_chi2), "p_value": float(srm_p)}

# balance of covariates across arms (randomization check)
balance = {}
for c in ["channel", "platform", "country"]:
    ct = pd.crosstab(um[c], um["variant"])
    balance[c] = float(stats.chi2_contingency(ct)[1])


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def prop_test(df, col):
    a, b = df[df.variant == "A"][col], df[df.variant == "B"][col]
    nA, nB, sA, sB = len(a), len(b), a.sum(), b.sum()
    pA, pB = sA / nA, sB / nB
    se = np.sqrt(pA * (1 - pA) / nA + pB * (1 - pB) / nB)
    diff = pB - pA
    _, p = proportions_ztest([sB, sA], [nB, nA])
    return dict(n_A=int(nA), n_B=int(nB), rate_A=pA, rate_B=pB, diff=diff,
                ci_low=diff - Z * se, ci_high=diff + Z * se,
                rel_lift=diff / pA if pA > 0 else np.nan, p_value=float(p))


def boot_diff(a, b, n_iter=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.empty(n_iter)
    for i in range(n_iter):
        out[i] = rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean()
    return out


def mean_test(df, col, n_iter=2000):
    a, b = df[df.variant == "A"][col].to_numpy(), df[df.variant == "B"][col].to_numpy()
    d = boot_diff(a, b, n_iter)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return dict(mean_A=float(a.mean()), mean_B=float(b.mean()), diff=float(b.mean() - a.mean()),
                ci_low=float(lo), ci_high=float(hi), p_value=float(max(p, 1 / n_iter)), boot=d)


def mde(p, n, alpha=0.05, power=0.8):
    return (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * np.sqrt(2 * p * (1 - p) / n)


# ----------------------------------------------------------------------------
# 3. Primary metric: retention
# ----------------------------------------------------------------------------
ret = {m: prop_test(um, m) for m in ["d1", "d7", "d14"]}

# ----------------------------------------------------------------------------
# 4. Engagement + guardrails
# ----------------------------------------------------------------------------
r_a = um[um.variant == "A"]["rounds_7d"]
r_b = um[um.variant == "B"]["rounds_7d"]
mw = stats.mannwhitneyu(r_b, r_a, alternative="two-sided")
engagement = {
    "rounds_7d_mean_A": float(r_a.mean()), "rounds_7d_mean_B": float(r_b.mean()),
    "rounds_7d_median_A": float(r_a.median()), "rounds_7d_median_B": float(r_b.median()),
    "mann_whitney_p": float(mw.pvalue),
}
payer = prop_test(um, "payer")
arpu = mean_test(um, "revenue_14d")
payers_only = um[um.payer == 1]
arppu = mean_test(payers_only, "revenue_14d", n_iter=2000)

monet = {"payer_rate": payer,
         "arpu": {k: v for k, v in arpu.items() if k != "boot"},
         "arppu": {k: v for k, v in arppu.items() if k != "boot"}}

# ----------------------------------------------------------------------------
# 5. Heterogeneity
# ----------------------------------------------------------------------------
seg_rows = []
for dim in ["channel", "platform"]:
    for lvl in sorted(um[dim].unique()):
        sub = um[um[dim] == lvl]
        d7 = prop_test(sub, "d7")
        d1 = prop_test(sub, "d1")
        ar = mean_test(sub, "revenue_14d", n_iter=1000)
        py = prop_test(sub, "payer")
        d7_gain = d7["diff"]
        arpu_delta = ar["diff"]
        seg_rows.append(dict(
            dimension=dim, segment=lvl, n=len(sub),
            d1_diff_pp=d1["diff"] * 100,
            d7_A=d7["rate_A"], d7_B=d7["rate_B"], d7_diff_pp=d7_gain * 100,
            d7_ci_low_pp=d7["ci_low"] * 100, d7_ci_high_pp=d7["ci_high"] * 100, d7_p=d7["p_value"],
            payer_diff_pp=py["diff"] * 100, payer_p=py["p_value"],
            arpu_A=ar["mean_A"], arpu_B=ar["mean_B"], arpu_diff=arpu_delta,
            arpu_ci_low=ar["ci_low"], arpu_ci_high=ar["ci_high"], arpu_p=ar["p_value"],
            extra_d7_users_per_10k=d7_gain * 10_000,
            arpu_delta_per_10k_usd=arpu_delta * 10_000,
            # extra future revenue an additionally retained D7 user must generate (beyond day 14)
            # for B to pay for its ARPU cost; 0 if no ARPU cost
            breakeven_value_per_extra_d7_user=(max(0.0, -arpu_delta) / d7_gain) if d7_gain > 0 else np.nan,
        ))
seg = pd.DataFrame(seg_rows)
seg["d7_p_bh"] = multipletests(seg["d7_p"], method="fdr_bh")[1]
seg["arpu_p_bh"] = multipletests(seg["arpu_p"], method="fdr_bh")[1]
seg.to_csv(RES / "segment_table.csv", index=False)

# regression: adjusted effect + interaction test
m0 = smf.logit("d7 ~ C(variant) + C(channel) + C(platform) + C(country)", um).fit(disp=0)
m1 = smf.logit("d7 ~ C(variant) * C(channel) + C(platform) + C(country)", um).fit(disp=0)
lr = 2 * (m1.llf - m0.llf)
lr_df = m1.df_model - m0.df_model
reg = {
    "odds_ratio_B": float(np.exp(m0.params["C(variant)[T.B]"])),
    "or_ci_low": float(np.exp(m0.conf_int().loc["C(variant)[T.B]", 0])),
    "or_ci_high": float(np.exp(m0.conf_int().loc["C(variant)[T.B]", 1])),
    "interaction_lr_chi2": float(lr), "interaction_df": int(lr_df),
    "interaction_p": float(stats.chi2.sf(lr, lr_df)),
}

# ----------------------------------------------------------------------------
# 6. Power / MDE
# ----------------------------------------------------------------------------
n_arm = int(min(cnt))
power = {
    "mde_d7_overall_pp": float(mde(ret["d7"]["rate_A"], n_arm) * 100),
    "mde_d7_by_segment_pp": {
        r.segment: float(mde(r.d7_A, r.n / 2) * 100) for r in seg.itertuples() if r.dimension == "channel"
    },
}

# ----------------------------------------------------------------------------
# 7. Figures
# ----------------------------------------------------------------------------
def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return lo, hi


# --- Fig 1: retention curve + lifts
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
for v, col in [("A", COL_A), ("B", COL_B)]:
    c = curve[curve.variant == v].sort_values("day_n")
    lo, hi = wilson(c.retained.to_numpy(), c.cohort_size.to_numpy())
    ax[0].plot(c.day_n, c.retention * 100, marker="o", ms=3.5, color=col, label=f"Variant {v}" + (" (Reward Pack)" if v == "B" else " (control)"))
    ax[0].fill_between(c.day_n, lo * 100, hi * 100, color=col, alpha=0.18)
ax[0].set_xlim(1, 14)
ax[0].set_ylim(bottom=0)
c1 = curve[curve.day_n >= 1]
ax[0].set_ylim(0, c1.retention.max() * 100 * 1.12)
ax[0].set_xlabel("Days since install")
ax[0].set_ylabel("Retention (%)")
ax[0].set_title("Retention curve by variant (95% CI)")
ax[0].legend(frameon=False)

labels, diffs, los, his, sig = [], [], [], [], []
for m, lab in [("d1", "D1"), ("d7", "D7"), ("d14", "D14")]:
    r = ret[m]
    labels.append(lab)
    diffs.append(r["diff"] * 100)
    los.append(r["diff"] * 100 - r["ci_low"] * 100)
    his.append(r["ci_high"] * 100 - r["diff"] * 100)
    sig.append(r["p_value"] < 0.05)
ax[1].bar(labels, diffs, yerr=[los, his], capsize=5, color=[COL_B if s else COL_A for s in sig], alpha=0.9)
for i, (d, r) in enumerate(zip(diffs, [ret["d1"], ret["d7"], ret["d14"]])):
    ax[1].text(i, d + his[i] + 0.05, f"{d:+.2f}pp\np={r['p_value']:.3g}", ha="center", va="bottom", fontsize=8.5)
ax[1].axhline(0, color=COL_GREY, lw=0.8)
ax[1].set_ylabel("B minus A (percentage points)")
ax[1].set_title("Retention lift (95% CI)")
ax[1].set_ylim(min(0, min(diffs) - max(los)) - 0.2, max(d + h for d, h in zip(diffs, his)) + 0.9)
plt.tight_layout()
plt.savefig(FIG / "fig1_retention.png")
plt.close()

# --- Fig 2: segment forest plots
overall_row = dict(dimension="overall", segment="Overall", n=len(um),
                   d7_diff_pp=ret["d7"]["diff"] * 100, d7_ci_low_pp=ret["d7"]["ci_low"] * 100,
                   d7_ci_high_pp=ret["d7"]["ci_high"] * 100, d7_p_bh=ret["d7"]["p_value"],
                   arpu_diff=arpu["diff"], arpu_ci_low=arpu["ci_low"], arpu_ci_high=arpu["ci_high"],
                   arpu_p_bh=arpu["p_value"])
fp = pd.concat([pd.DataFrame([overall_row]), seg], ignore_index=True)
ypos = np.arange(len(fp))[::-1]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for i, r in fp.iterrows():
    y = ypos[i]
    s7 = r.d7_p_bh < 0.05
    ax[0].errorbar(r.d7_diff_pp, y, xerr=[[r.d7_diff_pp - r.d7_ci_low_pp], [r.d7_ci_high_pp - r.d7_diff_pp]],
                   fmt="o", color=COL_B if s7 else COL_A, capsize=3, ms=6 if r.segment != "Overall" else 8)
    sa = r.arpu_p_bh < 0.05
    ax[1].errorbar(r.arpu_diff, y, xerr=[[r.arpu_diff - r.arpu_ci_low], [r.arpu_ci_high - r.arpu_diff]],
                   fmt="s", color=COL_B if sa else COL_A, capsize=3, ms=6 if r.segment != "Overall" else 8)
ylabels = [f"{r.segment}  (n={int(r.n):,})" for r in fp.itertuples()]
ax[0].set_yticks(ypos)
ax[0].set_yticklabels(ylabels)
for a_, t, xl in [(ax[0], "D7 retention lift by segment (pp, 95% CI)", "B minus A (percentage points)"),
                  (ax[1], "14-day ARPU change by segment ($, 95% bootstrap CI)", "B minus A (USD per user)")]:
    a_.axvline(0, color=COL_GREY, lw=0.8)
    a_.set_title(t, fontsize=10)
    a_.set_xlabel(xl)
ax[0].axhline(ypos[0] - 0.5, color="#bbbbbb", lw=0.8)
ax[1].axhline(ypos[0] - 0.5, color="#bbbbbb", lw=0.8)
fig.text(0.5, -0.01, "Orange = significant (segments: Benjamini-Hochberg-adjusted, FDR 5%; Overall: unadjusted); grey = not significant", ha="center", fontsize=8.5)
plt.tight_layout()
plt.savefig(FIG / "fig2_segments_forest.png", bbox_inches="tight")
plt.close()

# --- Fig 3: monetization guardrails
fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
pr = [payer["rate_A"] * 100, payer["rate_B"] * 100]
pa = um[um.variant == "A"]["payer"]
pb = um[um.variant == "B"]["payer"]
lo_a, hi_a = wilson(pa.sum(), len(pa))
lo_b, hi_b = wilson(pb.sum(), len(pb))
ax[0].bar(["A", "B"], pr, color=[COL_A, COL_B], yerr=[[pr[0] - lo_a * 100, pr[1] - lo_b * 100], [hi_a * 100 - pr[0], hi_b * 100 - pr[1]]], capsize=5)
ax[0].set_title(f"Payer rate, 14d  (p={payer['p_value']:.3g})")
ax[0].set_ylabel("% of users who purchased")
ax[0].set_ylim(0, max(pr) * 1.35)
for i, v in enumerate(pr):
    ax[0].text(i, [hi_a, hi_b][i] * 100 + 0.1, f"{v:.2f}%", ha="center", fontsize=9)

arpu_vals = [arpu["mean_A"], arpu["mean_B"]]
ab = um[um.variant == "A"]["revenue_14d"].to_numpy()
bb = um[um.variant == "B"]["revenue_14d"].to_numpy()
se_a, se_b = ab.std(ddof=1) / np.sqrt(len(ab)), bb.std(ddof=1) / np.sqrt(len(bb))
ax[1].bar(["A", "B"], arpu_vals, color=[COL_A, COL_B], yerr=[Z * se_a, Z * se_b], capsize=5)
ax[1].set_title(f"ARPU, 14d  (p={arpu['p_value']:.3g})")
ax[1].set_ylabel("USD per installed user")
for i, v in enumerate(arpu_vals):
    ax[1].text(i, v + Z * [se_a, se_b][i] + 0.005, f"${v:.3f}", ha="center", fontsize=9)
ax[1].set_ylim(0, max(arpu_vals) * 1.45)

ax[2].hist(arpu["boot"], bins=40, color=COL_A, alpha=0.8)
ax[2].axvline(0, color=COL_GREY, lw=1)
ax[2].axvline(arpu["diff"], color=COL_B, lw=2, label=f"observed {arpu['diff']:+.3f}")
ax[2].axvline(arpu["ci_low"], color=COL_B, lw=1, ls="--")
ax[2].axvline(arpu["ci_high"], color=COL_B, lw=1, ls="--", label="95% CI")
ax[2].set_title("Bootstrap: ARPU difference (B-A)")
ax[2].set_xlabel("USD per user")
ax[2].legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig3_monetization.png")
plt.close()

# --- Fig 4: data quality + engagement
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
bins = np.logspace(np.log10(max(avg_rounds.min(), 1)), np.log10(avg_rounds.max()), 60)
ax[0].hist(human_avg, bins=bins, color=COL_A, alpha=0.85, label="Kept")
ax[0].hist(avg_rounds[avg_rounds.index.isin(bot_ids)], bins=bins, color=COL_B, alpha=0.9, label="Removed as bot-like")
ax[0].axvline(150, color=COL_GREY, ls="--", lw=1)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel("Average rounds per active day (log)")
ax[0].set_ylabel("Users (log)")
ax[0].set_title("Data quality: bot-like accounts")
ax[0].legend(frameon=False)

mx = np.percentile(um.rounds_7d, 99)
bins2 = np.linspace(0, mx, 50)
ax[1].hist(r_a.clip(upper=mx), bins=bins2, alpha=0.55, color=COL_A, label=f"A (median {r_a.median():.0f})", density=True)
ax[1].hist(r_b.clip(upper=mx), bins=bins2, alpha=0.55, color=COL_B, label=f"B (median {r_b.median():.0f})", density=True)
ax[1].set_xlabel("Rounds played in first 7 days (clipped at p99)")
ax[1].set_ylabel("Density")
ax[1].set_title(f"Early engagement  (Mann-Whitney p={engagement['mann_whitney_p']:.3g})")
ax[1].legend(frameon=False)
plt.tight_layout()
plt.savefig(FIG / "fig4_quality_engagement.png")
plt.close()

# --- Fig 5: trade-off by channel
ch = seg[seg.dimension == "channel"]
fig, ax = plt.subplots(figsize=(6.4, 5))
ax.scatter(ch.d7_diff_pp, ch.arpu_diff, s=ch.n / 25, color=COL_B, alpha=0.65, edgecolor="white")
ax.errorbar(ch.d7_diff_pp, ch.arpu_diff,
            xerr=[ch.d7_diff_pp - ch.d7_ci_low_pp, ch.d7_ci_high_pp - ch.d7_diff_pp],
            yerr=[ch.arpu_diff - ch.arpu_ci_low, ch.arpu_ci_high - ch.arpu_diff],
            fmt="none", ecolor="#999999", lw=0.8, zorder=0)
for r in ch.itertuples():
    ax.annotate(r.segment, (r.d7_diff_pp, r.arpu_diff), textcoords="offset points", xytext=(8, 8), fontsize=9)
ax.axhline(0, color=COL_GREY, lw=0.8)
ax.axvline(0, color=COL_GREY, lw=0.8)
ax.set_xlabel("D7 retention lift (pp)  ->  more engagement")
ax.set_ylabel("14-day ARPU change (USD)  ->  more revenue")
ax.set_title("Trade-off by acquisition channel\n(bubble size = users; bars = 95% CI)")
plt.tight_layout()
plt.savefig(FIG / "fig5_tradeoff.png")
plt.close()

# ----------------------------------------------------------------------------
# summary output
# ----------------------------------------------------------------------------
summary = dict(data_quality=dq, srm=srm, covariate_balance_p=balance, retention=ret,
               engagement=engagement, monetization=monet, regression=reg, power=power)
(RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float, ensure_ascii=False))

pd.set_option("display.width", 200, "display.max_columns", 30)
print("=== DATA QUALITY ===");  print(json.dumps(dq, indent=1))
print("\n=== SRM ===", srm, "\nbalance p:", balance)
print("\n=== RETENTION ===")
for k, v in ret.items():
    print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
print("\n=== ENGAGEMENT ===", engagement)
print("\n=== MONETIZATION ===")
for k, v in monet.items():
    print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
print("\n=== REGRESSION ===", reg)
print("\n=== POWER ===", power)
print("\n=== SEGMENTS ===")
print(seg.round(4).to_string(index=False))
