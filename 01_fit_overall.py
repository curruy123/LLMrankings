"""
01_fit_overall.py — Step 1: data checks + overall BTL ranking.

Run from the project folder:
    python 01_fit_overall.py

Outputs (in ./outputs/):
    overall_ranking.csv          ranking with model-based and session-robust SEs,
                                 bootstrap rank intervals, P(rank 1)
    overall_bootstrap_ranks.csv  rank of every model in each bootstrap replicate
    overall_fit_summary.csv      fit-level diagnostics (n, gamma, LR test, ...)

Changes vs. the first version: exact Newton solver, identified covariance,
session-clustered SEs, MLE-existence diagnostics, Wald + LR test for gamma,
and a session-cluster bootstrap of the ranks.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, spearmanr

import btl
from common import (LANGUAGE, MIN_COMPARISONS, N_BOOT, OUT, SEED, load_raw,
                    section)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

# ---------------- 1. load + verify ----------------
raw = load_raw()
section("RAW FILE CHECKS")
print("rows:", len(raw))
print("unique models (all languages):",
      pd.concat([raw.model_a, raw.model_b]).nunique())
print("languages present:", raw["language"].nunique(),
      "(top:", raw["language"].value_counts().head(3).to_dict(), ")")
print("self-matches (model_a == model_b):", int((raw.model_a == raw.model_b).sum()))

df = raw if LANGUAGE is None else raw[raw["language"] == LANGUAGE]
print(f"after language filter ({LANGUAGE}):", len(df))

dec = btl.prepare_decisive(df)
print("decisive comparisons used for fitting:", len(dec))
print("dropped ties:", int((df.winner == "tie").sum()),
      " dropped both_bad:", int((df.winner == "both_bad").sum()),
      f" ({100 * (1 - len(dec) / len(df)):.1f}% of {LANGUAGE} rows)")
print("evaluation sessions:", dec.evaluation_session_id.nunique(),
      "| share of sessions with >1 vote:",
      round((dec.groupby('evaluation_session_id').size() > 1).mean(), 3))

# ---------------- 2. coverage + connectivity ----------------
section("COVERAGE")
counts = btl.model_counts(dec)
print("models:", len(counts), "| min / median comparisons:",
      counts.min(), "/", int(counts.median()))
print("models with <", MIN_COMPARISONS, "comparisons:")
print(counts[counts < MIN_COMPARISONS].to_string() or "  none")
missing = set(pd.concat([raw.model_a, raw.model_b])) - set(counts.index)
print("models present in full data but absent here:", sorted(missing) or "none")

models = sorted(counts.index)
n_comp, labels, n_strong = btl.comparison_graph(dec, models)
pairs = dec[["model_a", "model_b"]].apply(lambda r: tuple(sorted(r)), axis=1)
section("COMPARISON GRAPH")
print("connected components (undirected):", n_comp)
print("strongly connected components (win graph):", n_strong,
      " <- 1 means the MLE exists for every model")
print(f"observed model pairs: {pairs.nunique()} of {len(models) * (len(models) - 1) // 2}")

# ---------------- 3. fit ----------------
section("FIT")
fit_pos = btl.fit_btl(dec, models, position_effect=True)
fit_nopos = btl.fit_btl(dec, models, position_effect=False)
print("converged:", fit_pos["converged"], fit_nopos["converged"],
      "| Newton iterations:", fit_pos["n_iter"],
      "| Hessian condition number:", round(fit_pos["hessian_condition"], 1))
g, gse = fit_pos["gamma"], fit_pos["gamma_se_robust"]
print(f"position effect gamma = {g:.4f} (robust SE {gse:.4f}, z = {g / gse:.2f}, "
      f"p = {2 * norm.sf(abs(g / gse)):.3f})  [positive = position A favoured]")
print(f"  -> odds multiplier for position A: {np.exp(g):.3f}; "
      f"P(A wins | equal models) = {1 / (1 + np.exp(-g)):.4f}")
lr = 2 * (fit_pos["loglik"] - fit_nopos["loglik"])
print(f"likelihood-ratio stat for gamma: {lr:.2f} (chi2_1 p = {chi2.sf(lr, 1):.3f})")
same = (fit_pos["theta"].rank() == fit_nopos["theta"].rank()).mean()
print(f"share of models with identical rank with/without gamma: {same:.3f}")
r = fit_pos["se_robust"] / fit_pos["se"]
print(f"session-robust / model-based SE ratio: mean {r.mean():.3f}, max {r.max():.3f}")

# ---------------- 4. ranking + bootstrap ----------------
table = btl.ranking_table(fit_pos, dec)
table["low_data"] = table["n_comparisons"] < MIN_COMPARISONS
print(f"\nrunning {N_BOOT} session-cluster bootstrap refits for rank uncertainty ...")
boot = btl.bootstrap_ranks(dec, models, B=N_BOOT, seed=SEED)
table = table.join(btl.rank_summary(boot))
boot["ranks"].to_csv(OUT / "overall_bootstrap_ranks.csv", index=False)
section("OVERALL BTL RANKING")
cols = ["rank", "theta", "se_robust", "ci_low_robust", "ci_high_robust", "elo",
        "n_comparisons", "raw_win_rate", "rank_ci_low", "rank_ci_high", "p_rank1"]
print(table[cols].round(3).to_string())

top = table.index[0]
second = table.index[1]
d, s = btl.diff_se(fit_pos, top, second)
print(f"\n#1 vs #2: {top} - {second}: diff {d:.3f} (SE {s:.3f}, z {d / s:.2f}); "
      f"P(win) = {1 / (1 + np.exp(-d)):.3f}")

# ---------------- 5. BTL vs raw win rate ----------------
section("BTL vs RAW WIN RATE")
rho, _ = spearmanr(table["theta"], table["raw_win_rate"])
print(f"Spearman correlation of BTL strength with raw win rate: {rho:.4f}")
table["raw_rank"] = table["raw_win_rate"].rank(ascending=False).astype(int)
table["rank_shift"] = table["raw_rank"] - table["rank"]
print("Largest rank changes after adjusting for opponent strength:")
print(table.reindex(table["rank_shift"].abs().sort_values(ascending=False).index)
      [["rank", "raw_rank", "rank_shift", "raw_win_rate", "n_comparisons"]]
      .head(8).round(3).to_string())

table.to_csv(OUT / "overall_ranking.csv")
pd.DataFrame([{
    "n_rows_language": len(df), "n_decisive": len(dec), "n_models": len(models),
    "n_sessions": dec.evaluation_session_id.nunique(),
    "undirected_components": n_comp, "strong_components": n_strong,
    "gamma": g, "gamma_se": fit_pos["gamma_se"], "gamma_se_robust": gse,
    "lr_gamma": lr, "p_lr_gamma": chi2.sf(lr, 1), "spearman_btl_raw": rho,
    "top_model": top, "top_p_rank1": table.loc[top, "p_rank1"],
    "converged": fit_pos["converged"], "bootstrap_not_strongly_connected":
        boot["not_strongly_connected"],
}]).to_csv(OUT / "overall_fit_summary.csv", index=False)
print("\nSaved:", OUT / "overall_ranking.csv", "|", OUT / "overall_fit_summary.csv")
