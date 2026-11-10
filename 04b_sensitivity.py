"""
04b_sensitivity.py — Part E: robustness / sensitivity analyses.

Run:  python 04b_sensitivity.py   (after 01–04; uses lambda chosen in 04)

Each analysis refits the model under an alternative choice and reports how
much the ranking (Spearman with the main ranking), the #1 model and the top-3
change. Covered:
  S1  ties and both_bad scored as half-wins (vs excluded)
  S2  position effect removed
  S3  unequal matchup frequencies: every observed model pair weighted equally
  S4  overlapping categories: category fits on category-EXCLUSIVE rows
  S5  within-session correlation: one randomly chosen vote per session
  S6  English-only restriction: all languages
  S7  sparse models: ridge-regularised category fits (l2 = 1) and the
      partial-pooling ("shrunk") category ranking on all data
  S8  one-dimensional strength: pair-level goodness-of-fit of the overall BTL
  S9  unequal numbers of comparisons per model: rank-interval width vs n

Outputs: outputs/sensitivity_summary.csv, sensitivity_tie_rates.csv,
         sensitivity_gof_pairs.csv, shrunk_<cat>_ranking.csv,
         figures/gof_pairs.png, figures/rank_ci_vs_n.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import chi2, spearmanr

import btl
from common import (CATEGORIES, COLORS, FIG, INK, INK2, OUT, SEED, load_raw,
                    section)

pd.set_option("display.width", 220)
raw = load_raw()
en = raw[raw.language == "en"]
dec = btl.prepare_decisive(en)
rng = np.random.default_rng(SEED)
main = {"overall": pd.read_csv(OUT / "overall_ranking.csv", index_col=0)}
for c in CATEGORIES:
    main[c] = pd.read_csv(OUT / f"{c}_ranking.csv", index_col=0)

rows = []


def compare(analysis, scope, fit, note=""):
    ref = main[scope]
    th = fit["theta"]
    common = ref.index.intersection(th.index)
    rho = spearmanr(ref.loc[common, "theta"], th[common])[0]
    rel = ref.index[ref.get("reliable", pd.Series(True, index=ref.index))].intersection(th.index)
    rho_rel = spearmanr(ref.loc[rel, "theta"], th[rel])[0]
    top_new = th.sort_values(ascending=False).index[:3].tolist()
    top_ref = ref.index[:3].tolist()
    rows.append({"analysis": analysis, "scope": scope, "n": fit["n"], "spearman_vs_main": rho,
                 "spearman_vs_main_reliable": rho_rel, "top1_main": top_ref[0], "top1_new": top_new[0],
                 "top1_same": top_ref[0] == top_new[0],
                 "top3_overlap": len(set(top_ref) & set(top_new)),
                 "gamma": fit["gamma"], "note": note})
    print(f"  {analysis:38s} {scope:17s} n={fit['n']:6d}  rho={rho:.3f} (reliable {rho_rel:.3f})  "
          f"#1: {top_new[0]}{'' if top_ref[0] == top_new[0] else '  <-- CHANGED'}")


def scopes(d):
    yield "overall", d
    for c in CATEGORIES:
        yield c, d[d[c]]


# ---------------- S1 ties ----------------
section("S1  TIES / BOTH_BAD AS HALF-WINS")
both = btl.prepare_with_ties(en, include_both_bad=True)
ties_only = btl.prepare_with_ties(en, include_both_bad=False)
for scope, d in scopes(both):
    compare("ties+both_bad = 0.5", scope, btl.fit_btl(d, cluster=None))
for scope, d in scopes(ties_only):
    compare("ties = 0.5 (both_bad dropped)", scope, btl.fit_btl(d, cluster=None))
# are tie rates related to strength? (if so, excluding ties is not innocuous)
e2 = en[en.model_a != en.model_b]
long = pd.concat([e2[["model_a", "winner"]].rename(columns={"model_a": "m"}),
                  e2[["model_b", "winner"]].rename(columns={"model_b": "m"})])
tr = long.groupby("m").winner.agg(tie_rate=lambda s: (s == "tie").mean(),
                                  both_bad_rate=lambda s: (s == "both_bad").mean(), n="size")
tr = tr.join(main["overall"][["theta", "rank"]], how="inner")
r_tie = spearmanr(tr.theta, tr.tie_rate)[0]
r_bb = spearmanr(tr.theta, tr.both_bad_rate)[0]
print(f"Spearman(theta, tie rate) = {r_tie:.3f}; Spearman(theta, both_bad rate) = {r_bb:.3f}")
print(f"overall tie share {(en.winner == 'tie').mean():.3f}, both_bad share {(en.winner == 'both_bad').mean():.3f}")
tr.sort_values("rank").to_csv(OUT / "sensitivity_tie_rates.csv")

# ---------------- S2 position ----------------
section("S2  NO POSITION EFFECT")
for scope, d in scopes(dec):
    compare("no position effect", scope, btl.fit_btl(d, position_effect=False, cluster=None))
print(f"share of decisive votes won by position A: {dec.y.mean():.4f} (n = {len(dec)})")

# ---------------- S3 matchup frequency ----------------
section("S3  EQUAL WEIGHT PER OBSERVED MODEL PAIR")
for scope, d in scopes(dec):
    pair = d.model_a.where(d.model_a < d.model_b, d.model_b) + "|" + d.model_b.where(d.model_a < d.model_b, d.model_a)
    w = 1.0 / pair.map(pair.value_counts()).to_numpy()
    w *= len(d) / w.sum()
    compare("equal weight per pair", scope, btl.fit_btl(d, weights=w, cluster=None),
            note=f"pair counts: median {int(pair.value_counts().median())}, max {pair.value_counts().max()}")

# ---------------- S4 overlap ----------------
section("S4  CATEGORY-EXCLUSIVE ROWS")
cats = list(CATEGORIES)
for c in cats:
    others = [o for o in cats if o != c]
    d = dec[dec[c] & ~dec[others].any(axis=1)]
    compare("category-exclusive rows only", c, btl.fit_btl(d, cluster=None),
            note=f"dropped {int((dec[c] & dec[others].any(axis=1)).sum())} overlapping rows")

# ---------------- S5 sessions ----------------
section("S5  ONE VOTE PER SESSION")
sizes = dec.groupby("evaluation_session_id").size()
print(f"sessions: {len(sizes)}; share with >1 decisive vote: {(sizes > 1).mean():.3f}; "
      f"share of rows in multi-vote sessions: {sizes[sizes > 1].sum() / len(dec):.3f}; max {sizes.max()}")
one = dec.sample(frac=1, random_state=SEED).drop_duplicates("evaluation_session_id")
for scope, d in scopes(one):
    compare("one random vote per session", scope, btl.fit_btl(d, cluster=None))
f = btl.fit_btl(dec)
r = f["se_robust"] / f["se"]
print(f"robust/model SE ratio overall: mean {r.mean():.3f}, max {r.max():.3f}")

# ---------------- S6 language ----------------
section("S6  ALL LANGUAGES")
alld = btl.prepare_decisive(raw)
for scope, d in scopes(alld):
    compare("all languages", scope, btl.fit_btl(d, cluster=None),
            note="includes qwen3-coder-480b (absent from English)" if scope == "overall" else "")

# ---------------- S7 sparse models ----------------
section("S7  REGULARISED / PARTIAL-POOLING CATEGORY RANKINGS")
fo = btl.fit_btl(dec, cluster=None)
lam_cv = pd.read_csv(OUT / "prediction_split_info_cv.csv").groupby("category")["lambda"].median()
for c in CATEGORIES:
    d = dec[dec[c]]
    compare("ridge l2 = 1", c, btl.fit_btl(d, l2=1.0, cluster=None))
    lam = float(lam_cv[c])
    off = d.model_a.map(fo["theta"]).to_numpy() - d.model_b.map(fo["theta"]).to_numpy()
    fs = btl.fit_btl(d, fo["models"], l2=lam, offset=off, cluster=None)
    comb = dict(fs)
    comb["theta"] = fo["theta"] + fs["theta"]
    comb["theta"] -= comb["theta"].mean()
    compare(f"partial pooling (lambda = {lam:g})", c, comb)
    t = pd.DataFrame({"theta_shrunk": comb["theta"], "delta": fs["theta"],
                      "theta_overall": fo["theta"]}).sort_values("theta_shrunk", ascending=False)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    t["category_only_rank"] = main[c]["rank"].reindex(t.index)
    t["overall_rank"] = main["overall"]["rank"].reindex(t.index)
    t.to_csv(OUT / f"shrunk_{c}_ranking.csv")
    print(t.head(5).round(3).to_string())

# ---------------- S8 goodness of fit ----------------
section("S8  PAIR-LEVEL GOODNESS OF FIT (ONE-DIMENSIONAL STRENGTH)")
d = dec.copy()
d["p"] = btl.predict_proba(f, d)
key = np.where(d.model_a < d.model_b, d.model_a + "|" + d.model_b, d.model_b + "|" + d.model_a)
a_first = (d.model_a < d.model_b).to_numpy()
d["win_first"] = np.where(a_first, d.y, 1 - d.y)
d["p_first"] = np.where(a_first, d.p, 1 - d.p)
d["key"] = key
g = d.groupby("key").agg(n=("y", "size"), obs=("win_first", "sum"), exp=("p_first", "sum"),
                         var=("p_first", lambda s: (s * (1 - s)).sum()))
g = g[g.n >= 20]
g["z"] = (g.obs - g.exp) / np.sqrt(g["var"])
X2 = float((g.z ** 2).sum())
dfree = len(g) - (len(f["models"]) - 1) - 1
print(f"pairs with >= 20 comparisons: {len(g)}; Pearson X2 = {X2:.1f} on ~{dfree} df "
      f"(p = {chi2.sf(X2, dfree):.3g}); X2/df = {X2 / dfree:.3f}")
print(f"share of pairs with |z| > 1.96: {(g.z.abs() > 1.96).mean():.3f} (5% expected under BTL)")
g.sort_values("z").to_csv(OUT / "sensitivity_gof_pairs.csv")
rows.append({"analysis": "pair-level GOF (pairs n>=20)", "scope": "overall", "n": len(g),
             "note": f"X2={X2:.1f}, df~{dfree}, p={chi2.sf(X2, dfree):.3g}, "
                     f"share|z|>1.96={(g.z.abs() > 1.96).mean():.3f}"})

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(g.z, bins=40, density=True, color=COLORS["Overall"], alpha=0.75, edgecolor="white")
xs = np.linspace(-4.5, 4.5, 200)
ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.5, label="N(0,1) if BTL holds")
ax.set_xlabel("standardised residual per model pair (observed − expected wins)")
ax.set_ylabel("density")
ax.set_title(f"Overall BTL goodness of fit: {len(g)} pairs with ≥20 votes\n"
             f"X² / df = {X2 / dfree:.2f};  {100 * (g.z.abs() > 1.96).mean():.1f}% of |z| > 1.96 (5% expected)")
ax.legend(fontsize=8)
fig.savefig(FIG / "gof_pairs.png")
plt.close(fig)

# ---------------- S9 comparisons per model ----------------
section("S9  RANK UNCERTAINTY VS NUMBER OF COMPARISONS")
fig, ax = plt.subplots(figsize=(6.5, 4))
for scope, name, col in [("overall", "Overall", COLORS["Overall"])] + [
        (c, n, COLORS[n]) for c, n in CATEGORIES.items()]:
    t = main[scope]
    w = (t.rank_ci_high - t.rank_ci_low).astype(float)
    ax.scatter(t.n_comparisons, w, s=14, color=col, alpha=0.8, label=name)
    print(f"  {name:17s} Spearman(n, rank-CI width) = {spearmanr(t.n_comparisons, w)[0]:.3f}; "
          f"median width {w.median():.0f}")
ax.set_xscale("log")
ax.set_xlabel("decisive comparisons involving the model (log scale)")
ax.set_ylabel("width of 95% bootstrap rank interval")
ax.set_title("Fewer comparisons → much less certain rank")
ax.legend(fontsize=8)
fig.savefig(FIG / "rank_ci_vs_n.png")
plt.close(fig)

out = pd.DataFrame(rows)
out.to_csv(OUT / "sensitivity_summary.csv", index=False)
section("SUMMARY")
print(out.drop(columns=["note"]).round(3).to_string())
