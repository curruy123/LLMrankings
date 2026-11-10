"""
02_fit_categories.py — Part B: category-specific BTL models.

For each of math, coding and creative writing we fit a separate BTL model
(with position effect) to English decisive comparisons that carry the
category flag. Categories are NOT mutually exclusive: a row flagged both
math and coding is used in both fits (the overlap is reported).

Run:  python 02_fit_categories.py   (after 01_fit_overall.py)

Outputs (./outputs/):
    math_ranking.csv, coding_ranking.csv, creative_writing_ranking.csv
    category_fit_summary.csv     one row per category: n, models, graph, gamma
    category_overlap.csv         pairwise overlap of category flags
    category_top_models.csv      best model per category vs overall best
    <cat>_bootstrap_ranks.csv    bootstrap ranks (used by 03_compare_rankings.py)
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

import btl
from common import (CATEGORIES, MIN_COMPARISONS, N_BOOT, OUT, SEED,
                    load_decisive, section)

pd.set_option("display.width", 220)

dec = load_decisive()
overall = pd.read_csv(OUT / "overall_ranking.csv", index_col=0)
overall_best = overall.index[0]

# ---------------- overlap ----------------
section("CATEGORY OVERLAP (English decisive comparisons)")
cats = list(CATEGORIES)
ov = pd.DataFrame(index=cats, columns=cats, dtype=int)
for a in cats:
    for b in cats:
        ov.loc[a, b] = int((dec[a] & dec[b]).sum())
print(ov.to_string())
n_any = int(dec[cats].any(axis=1).sum())
n_multi = int((dec[cats].sum(axis=1) > 1).sum())
print(f"rows in >=1 category: {n_any}; in >1 category: {n_multi}; "
      f"in none: {len(dec) - n_any}")
for c in cats:
    sub = dec[dec[c]]
    others = [o for o in cats if o != c]
    print(f"  {c}: {len(sub)} rows, {100 * sub[others].any(axis=1).mean():.1f}% "
          f"also in another category")
ov.to_csv(OUT / "category_overlap.csv")

summary, tops = [], []
for c, name in CATEGORIES.items():
    section(f"{name.upper()}")
    sub = dec[dec[c]].copy()
    counts = btl.model_counts(sub)
    models = sorted(counts.index)
    n_comp, labels, n_strong, s_labels = btl.graph_diagnostics(sub, models)
    wl = btl.wins_losses(sub).reindex(models).fillna(0)
    unbeaten = wl.index[(wl.losses == 0) | (wl.wins == 0)].tolist()
    pairs = sub[["model_a", "model_b"]].apply(lambda r: tuple(sorted(r)), axis=1)
    print(f"comparisons: {len(sub)} | models: {len(models)} | observed pairs: "
          f"{pairs.nunique()} of {len(models) * (len(models) - 1) // 2}")
    print(f"comparisons per model: min {counts.min()}, median {int(counts.median())}, "
          f"max {counts.max()}")
    print(f"undirected components: {n_comp} | strongly connected components: {n_strong}")
    print("models with zero wins or zero losses:", unbeaten or "none")
    if n_strong > 1:
        # Report which models sit outside the largest strong component;
        # their unpenalised MLE does not exist.
        big = pd.Series(s_labels).value_counts().index[0]
        print("  models outside main strongly connected component:",
              [m for m, l in zip(models, s_labels) if l != big])

    fit = btl.fit_btl(sub, models, position_effect=True)
    fit0 = btl.fit_btl(sub, models, position_effect=False)
    lr = 2 * (fit["loglik"] - fit0["loglik"])
    g, gse = fit["gamma"], fit["gamma_se_robust"]
    print(f"converged: {fit['converged']} (iter {fit['n_iter']}), max |theta| "
          f"{fit['max_abs_theta']:.2f}, Hessian condition {fit['hessian_condition']:.0f}")
    print(f"position effect gamma = {g:.4f} (robust SE {gse:.4f}, p = "
          f"{2 * norm.sf(abs(g / gse)):.3f}); LR = {lr:.2f} (p = {chi2.sf(lr, 1):.3f})")

    table = btl.ranking_table(fit, sub)
    print(f"bootstrap: {N_BOOT} session-cluster refits ...")
    boot = btl.bootstrap_ranks(sub, models, B=N_BOOT, seed=SEED)
    boot["ranks"].to_csv(OUT / f"{c}_bootstrap_ranks.csv", index=False)
    table = table.join(btl.rank_summary(boot))
    table["rank_ci_width"] = (table.rank_ci_high - table.rank_ci_low).astype(float)
    # Reliability flags: explicit, never silent.
    table["low_data"] = table.n_comparisons < MIN_COMPARISONS
    table["zero_wins_or_losses"] = table.index.isin(unbeaten)
    table["wide_se"] = table.se_robust > 0.25          # +/-0.49 logits = +/-12 pp
    table["reliable"] = ~(table.low_data | table.zero_wins_or_losses | table.wide_se)
    table["overall_rank"] = overall["rank"].reindex(table.index)
    table.to_csv(OUT / f"{c}_ranking.csv")

    show = ["rank", "theta", "se_robust", "ci_low_robust", "ci_high_robust",
            "n_comparisons", "raw_win_rate", "rank_ci_low", "rank_ci_high",
            "p_rank1", "overall_rank", "reliable"]
    print(table[show].head(12).round(3).to_string())
    flagged = table[~table.reliable]
    print(f"\nmodels flagged as insufficient data ({len(flagged)}):")
    print(flagged[["n_comparisons", "se_robust", "low_data", "wide_se",
                   "zero_wins_or_losses"]].round(3).to_string() if len(flagged) else "  none")

    # ---- recommendation view: who is best, and is it the overall best? ----
    best = table.index[0]
    d, s = btl.diff_se(fit, best, overall_best)
    p_win = 1 / (1 + np.exp(-d))
    plausible = table.index[table.p_rank1 >= 0.05].tolist()
    print(f"\ncategory #1: {best} (P(rank 1) = {table.loc[best, 'p_rank1']:.2f}, "
          f"rank 95% CI {table.loc[best, 'rank_ci_low']}-{table.loc[best, 'rank_ci_high']})")
    print(f"overall #1 ({overall_best}) has category rank {table.loc[overall_best, 'rank']} "
          f"(P(rank 1) = {table.loc[overall_best, 'p_rank1']:.2f})")
    if best != overall_best:
        print(f"theta[{best}] - theta[{overall_best}] = {d:.3f} (SE {s:.3f}, z = {d / s:.2f}, "
              f"p = {2 * norm.sf(abs(d / s)):.3f}); implied P(category #1 beats overall #1) = {p_win:.3f}")
    print("models with P(rank 1) >= 5%:", plausible)

    summary.append({
        "category": c, "n_comparisons": len(sub), "n_models": len(models),
        "n_sessions": sub.evaluation_session_id.nunique(),
        "observed_pairs": pairs.nunique(),
        "min_comparisons_per_model": int(counts.min()),
        "median_comparisons_per_model": int(counts.median()),
        "undirected_components": n_comp, "strong_components": n_strong,
        "n_zero_wins_or_losses": len(unbeaten),
        "n_flagged_unreliable": int((~table.reliable).sum()),
        "gamma": g, "gamma_se_robust": gse, "gamma_p": 2 * norm.sf(abs(g / gse)),
        "lr_gamma": lr, "converged": fit["converged"],
        "bootstrap_not_strongly_connected": boot["not_strongly_connected"],
        "theta_sd": float(fit["theta"].std()),
    })
    best_rel = table.index[table.reliable][0]
    print(f"best among reliably estimated models: {best_rel} "
          f"(P(rank 1) = {table.loc[best_rel, 'p_rank1']:.2f})")
    tops.append({
        "category": c, "category_best": best,
        "category_best_reliable": bool(table.loc[best, "reliable"]),
        "category_best_n": int(table.loc[best, "n_comparisons"]),
        "best_reliable_model": best_rel,
        "category_best_p_rank1": table.loc[best, "p_rank1"],
        "category_best_rank_ci": f"{table.loc[best, 'rank_ci_low']}-{table.loc[best, 'rank_ci_high']}",
        "overall_best": overall_best,
        "overall_best_category_rank": int(table.loc[overall_best, "rank"]),
        "overall_best_p_rank1": table.loc[overall_best, "p_rank1"],
        "differs": best != overall_best,
        "theta_diff_best_minus_overall_best": d, "se_diff": s,
        "z": d / s if s > 0 else np.nan,
        "p_value": 2 * norm.sf(abs(d / s)) if s > 0 else np.nan,
        "implied_p_best_beats_overall_best": p_win,
        "models_p_rank1_ge_5pct": "; ".join(plausible),
    })

section("SUMMARY")
summary = pd.DataFrame(summary)
tops = pd.DataFrame(tops)
print(summary.round(4).to_string())
print(tops.round(3).to_string())
summary.to_csv(OUT / "category_fit_summary.csv", index=False)
tops.to_csv(OUT / "category_top_models.csv", index=False)
print("\nSaved category rankings and summaries to", OUT)
