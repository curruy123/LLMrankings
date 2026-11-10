"""
04_evaluate_predictions.py — Part D: do task-specific BTL models predict
held-out human preferences better than one overall BTL model, and do they
lead to better recommendations?

Run:  python 04_evaluate_predictions.py      (~10 min on a laptop)

DESIGN
------
Data: English decisive comparisons (same as 01–03).

Splits are GROUPED by evaluation_session_id: all votes from one session go to
the same side, so near-duplicate prompts / the same user never straddle
train and test.
  * Primary analysis: one 80/20 grouped split, fixed seed (SEED in common.py).
  * Robustness: 5 repetitions of grouped 5-fold cross-validation. Every row is
    held out exactly once per repetition (out-of-fold, "cross-fitted").

Predictors, all fitted on TRAINING rows only, all with their own position
effect gamma (consistent treatment of position bias):
  1. overall   : BTL fitted to all training comparisons (all tasks).
  2. category  : BTL fitted only to training comparisons in category c
                 (unpenalised MLE; if the training win graph is not strongly
                 connected, a small ridge l2 = 0.01 is used and recorded).
  3. shrunk    : partial pooling. logit P = theta_overall[a] - theta_overall[b]
                 + delta[a] - delta[b] + gamma_c, with ridge penalty
                 0.5*lambda*||delta||^2. lambda is chosen by an inner grouped
                 5-fold CV on the training data only (the overall offset is
                 refitted inside each inner fold, so no inner-validation row
                 informs its own prediction). lambda -> inf recovers `overall`,
                 lambda -> 0 recovers `category`.

Fallback for models absent from category training data:
  * category: rows where either model has no category training data are
    predicted by the overall model (the whole row, so that a prediction never
    mixes thetas from two separately centred fits). Counted and reported.
  * shrunk  : such models get delta = 0, i.e. their overall strength.
  * Rows involving a model absent from the OVERALL training data are dropped
    for all predictors (counted; in practice 0).
Every predictor is evaluated on exactly the same test rows in each category.
Test outcomes are never used for fitting or tuning.

Metrics: log loss (primary), accuracy, Brier score, calibration curves.
Uncertainty: paired session-cluster bootstrap (B = 2000) of the per-row
loss difference on the primary test set; spread across 25 CV folds.

RECOMMENDATION EVALUATION (training-based recommendations, held-out judges)
  R_overall  = top model of the overall training fit
  R_category = top model of the category training fit, among models with
               >= 50 category training comparisons and SE < 0.25 (an
               unreliable #1 would not be recommended in practice)
  R_shrunk   = top model under the shrunk fit
  (a) direct held-out battles between R_category and R_overall;
  (b) "disagreement" test rows, where the overall and category models
      disagree about which of the two LLMs is stronger: who agrees with the
      human more often?
  (c) held-out BTL "judge": a BTL model fitted ONLY to the held-out fold
      estimates theta(R_category) - theta(R_overall) using all indirect
      comparisons; combined across disjoint CV folds by averaging.

Outputs (./outputs/): prediction_metrics_primary.csv, prediction_diff_primary.csv,
prediction_metrics_cv.csv, prediction_diff_cv.csv, prediction_train_vs_test.csv,
recommendation_*.csv, figures/calibration.png, figures/prediction_diff.png
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit

import btl
from common import (CATEGORIES, COLORS, FIG, INK, INK2, MIN_COMPARISONS, OUT,
                    SEED, load_decisive, section)

LAMBDAS = [0.5, 2, 5, 10, 20, 50, 100, 300]
N_INNER = 5
N_OUTER = 5
N_REPEATS = 5
B_BOOT = 2000
METHODS = ["overall", "category", "shrunk"]
MLABEL = {"overall": "Overall", "category": "Category-only", "shrunk": "Shrunk (partial pooling)"}
pd.set_option("display.width", 220)

dec = load_decisive().reset_index(drop=True)
sess_codes, sess_uniq = pd.factorize(dec.evaluation_session_id)
G = len(sess_uniq)
ALL_MODELS = sorted(btl.model_counts(dec).index)
t0 = time.time()


def session_folds(codes_rows, n_folds, rng):
    """Assign each session (hence each row) to one of n_folds folds."""
    uniq = np.unique(codes_rows)
    perm = rng.permutation(len(uniq))
    fold_of = dict(zip(uniq[perm], np.arange(len(uniq)) % n_folds))
    return np.array([fold_of[c] for c in codes_rows])


def overall_offset(fit, df):
    th = fit["theta"]
    return df.model_a.map(th).to_numpy() - df.model_b.map(th).to_numpy()


def fit_category(train_c):
    models = sorted(btl.model_counts(train_c).index)
    _, _, ns = btl.comparison_graph(train_c, models)
    l2 = 0.0 if ns == 1 else 0.01
    f = btl.fit_btl(train_c, models, l2=l2, cluster="evaluation_session_id")
    f["ridge_needed"] = ns > 1
    return f


def fit_shrunk(train_c, fit_over, lam):
    off = overall_offset(fit_over, train_c)
    return btl.fit_btl(train_c, fit_over["models"], l2=lam, offset=off, cluster=None)


def tune_lambda(train, c, rng):
    """Inner grouped CV on the training data only."""
    folds = session_folds(sess_codes[train.index], N_INNER, rng)
    loss = np.zeros(len(LAMBDAS))
    nrows = 0
    for k in range(N_INNER):
        tr, va = train[folds != k], train[folds == k]
        va_c = va[va[c]]
        tr_c = tr[tr[c]]
        fo = btl.fit_btl(tr, sorted(btl.model_counts(tr).index), cluster=None)
        va_c = va_c[va_c.model_a.isin(fo["models"]) & va_c.model_b.isin(fo["models"])]
        off_va = overall_offset(fo, va_c)
        for j, lam in enumerate(LAMBDAS):
            fs = fit_shrunk(tr_c, fo, lam)
            p = btl.predict_proba(fs, va_c, offset=off_va)
            loss[j] += btl.log_loss_rows(va_c.y.to_numpy(), p).sum()
        nrows += len(va_c)
    return LAMBDAS[int(np.argmin(loss))], loss / nrows


def evaluate_split(train, test, rng, tag):
    """Fit all predictors on `train`, return row-level predictions on `test`."""
    fo = btl.fit_btl(train, sorted(btl.model_counts(train).index), cluster=None)
    rows, info = [], []
    for c in CATEGORIES:
        tr_c, te_c = train[train[c]], test[test[c]]
        keep = te_c.model_a.isin(fo["models"]) & te_c.model_b.isin(fo["models"])
        dropped = int((~keep).sum())
        te_c = te_c[keep]
        fc = fit_category(tr_c)
        lam, lam_curve = tune_lambda(train, c, rng)
        fs = fit_shrunk(tr_c, fo, lam)
        known = te_c.model_a.isin(fc["models"]) & te_c.model_b.isin(fc["models"])
        pred = pd.DataFrame({
            "row": te_c.index, "category": c, "split": tag,
            "session": sess_codes[te_c.index],
            "model_a": te_c.model_a.to_numpy(), "model_b": te_c.model_b.to_numpy(),
            "y": te_c.y.to_numpy(),
            "overall": btl.predict_proba(fo, te_c),
            "category_p": btl.predict_proba(fc, te_c, fallback=fo),
            "shrunk": btl.predict_proba(fs, te_c, offset=overall_offset(fo, te_c)),
            "fallback_used": (~known).to_numpy(),
        }).rename(columns={"category_p": "category_pred"})
        # Which model does each predictor think is stronger (ignoring position)?
        tho, thc = fo["theta"], fc["theta"]
        ths = fo["theta"] + fs["theta"]
        pred["fav_overall"] = np.where(te_c.model_a.map(tho).to_numpy() >= te_c.model_b.map(tho).to_numpy(), 1, 0)
        ca, cb = te_c.model_a.map(thc), te_c.model_b.map(thc)
        pred["fav_category"] = np.where(known, (ca.fillna(0) >= cb.fillna(0)).astype(int), pred.fav_overall)
        pred["fav_shrunk"] = (te_c.model_a.map(ths).to_numpy() >= te_c.model_b.map(ths).to_numpy()).astype(int)
        rows.append(pred)

        # training-set fit (in-sample) for the overfitting comparison
        tr_y = tr_c.y.to_numpy()
        tr_metrics = {
            "overall": btl.log_loss_rows(tr_y, btl.predict_proba(fo, tr_c)).mean(),
            "category": btl.log_loss_rows(tr_y, btl.predict_proba(fc, tr_c)).mean(),
            "shrunk": btl.log_loss_rows(tr_y, btl.predict_proba(
                fs, tr_c, offset=overall_offset(fo, tr_c))).mean(),
        }
        # recommendations from training data only
        n_tr = btl.model_counts(tr_c).reindex(fc["models"]).fillna(0)
        ok = (n_tr >= MIN_COMPARISONS) & (fc["se_robust"] < 0.25)
        rec = {
            "overall": fo["theta"].idxmax(),
            "category_raw": fc["theta"].idxmax(),
            "category": fc["theta"][ok].idxmax(),
            "shrunk": ths.idxmax(),
        }
        info.append({"split": tag, "category": c, "n_train": len(tr_c), "n_test": len(te_c),
                     "dropped_absent_overall": dropped,
                     "fallback_rows": int((~known).sum()),
                     "category_ridge_needed": fc["ridge_needed"], "lambda": lam,
                     **{f"train_logloss_{k}": v for k, v in tr_metrics.items()},
                     **{f"rec_{k}": v for k, v in rec.items()},
                     "gamma_overall": fo["gamma"], "gamma_category": fc["gamma"],
                     "gamma_shrunk": fs["gamma"]})
    return pd.concat(rows), pd.DataFrame(info)


def metric_table(pred):
    out = []
    for c, g in pred.groupby("category", sort=False):
        for m in METHODS:
            col = "category_pred" if m == "category" else m
            out.append({"category": c, "method": m, **btl.metrics(g.y.to_numpy(), g[col].to_numpy())})
    t = pd.DataFrame(out)
    macro = t.groupby("method", sort=False)[["log_loss", "accuracy", "brier"]].mean().reset_index()
    macro["category"], macro["n"] = "macro_average", t.groupby("method", sort=False).n.sum().to_numpy()
    pooled = []
    for m in METHODS:
        col = "category_pred" if m == "category" else m
        pooled.append({"category": "pooled_rows", "method": m,
                       **btl.metrics(pred.y.to_numpy(), pred[col].to_numpy())})
    return pd.concat([t, macro, pd.DataFrame(pooled)], ignore_index=True)


def paired_bootstrap(pred, rng, B=B_BOOT):
    """Session-cluster bootstrap CI for mean loss differences vs overall."""
    out = []
    groups = [(c, g) for c, g in pred.groupby("category", sort=False)] + [("pooled_rows", pred)]
    for c, g in groups:
        y = g.y.to_numpy()
        lo = btl.log_loss_rows(y, g.overall.to_numpy())
        codes, _ = pd.factorize(g.session)
        Gc = codes.max() + 1
        for m in ["category", "shrunk"]:
            col = "category_pred" if m == "category" else m
            p = g[col].to_numpy()
            for metric, d in [("log_loss", btl.log_loss_rows(y, p) - lo),
                              ("brier", (p - y) ** 2 - (g.overall.to_numpy() - y) ** 2),
                              ("accuracy", ((p > .5) == (y == 1)).astype(float) -
                               ((g.overall.to_numpy() > .5) == (y == 1)).astype(float))]:
                sd = np.bincount(codes, d, Gc)
                sn = np.bincount(codes, None, Gc)
                bs = np.empty(B)
                for b in range(B):
                    idx = rng.integers(0, Gc, Gc)
                    bs[b] = sd[idx].sum() / sn[idx].sum()
                out.append({"category": c, "method": m, "metric": metric,
                            "diff_vs_overall": d.mean(), "ci_low": np.quantile(bs, .025),
                            "ci_high": np.quantile(bs, .975),
                            "p_boot_two_sided": min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())),
                            "n": len(g)})
    return pd.DataFrame(out)


def disagreement(pred):
    """Accuracy on rows where two rankings disagree about the stronger model."""
    out = []
    for c, g in list(pred.groupby("category", sort=False)) + [("pooled_rows", pred)]:
        for other in ["category", "shrunk"]:
            dis = g[g.fav_overall != g[f"fav_{other}"]]
            if len(dis) == 0:
                out.append({"category": c, "vs": other, "n_disagree": 0})
                continue
            agree_other = (dis[f"fav_{other}"] == dis.y).mean()
            out.append({"category": c, "vs": other, "n_disagree": len(dis),
                        "share_of_test_rows": len(dis) / len(g),
                        f"human_sides_with_{other}": agree_other,
                        "human_sides_with_overall": 1 - agree_other,
                        "binomial_se": np.sqrt(agree_other * (1 - agree_other) / len(dis))})
    return pd.DataFrame(out)


# =====================================================================
# 1. PRIMARY 80/20 GROUPED SPLIT
# =====================================================================
section("PRIMARY GROUPED 80/20 SPLIT")
rng = np.random.default_rng(SEED)
test_sess = rng.random(G) < 0.2
is_test = test_sess[sess_codes]
train, test = dec[~is_test], dec[is_test]
print(f"train rows {len(train)}, test rows {len(test)}; sessions in both: "
      f"{len(set(train.evaluation_session_id) & set(test.evaluation_session_id))}")
pred_p, info_p = evaluate_split(train, test, rng, "primary")
mt = metric_table(pred_p)
diff_p = paired_bootstrap(pred_p, np.random.default_rng(SEED + 1))
dis_p = disagreement(pred_p)
print(info_p[["category", "n_train", "n_test", "fallback_rows", "dropped_absent_overall",
              "category_ridge_needed", "lambda", "rec_overall", "rec_category", "rec_shrunk"]].to_string())
print(mt.round(4).to_string())
print(diff_p.round(4).to_string())
print(dis_p.round(3).to_string())
mt.to_csv(OUT / "prediction_metrics_primary.csv", index=False)
diff_p.to_csv(OUT / "prediction_diff_primary.csv", index=False)
dis_p.to_csv(OUT / "recommendation_disagreement_primary.csv", index=False)
info_p.to_csv(OUT / "prediction_split_info_primary.csv", index=False)
pred_p.to_csv(OUT / "predictions_primary_test.csv", index=False)
print(f"[{time.time() - t0:.0f}s]")

# =====================================================================
# 2. REPEATED GROUPED 5-FOLD CROSS-VALIDATION
# =====================================================================
section(f"REPEATED GROUPED CV ({N_REPEATS} x {N_OUTER}-fold)")
cv_preds, cv_info = [], []
for r in range(N_REPEATS):
    rng_r = np.random.default_rng(SEED + 100 + r)
    folds = session_folds(sess_codes, N_OUTER, rng_r)
    for k in range(N_OUTER):
        p, i = evaluate_split(dec[folds != k], dec[folds == k], rng_r, f"r{r}f{k}")
        p["repeat"], i["repeat"], p["fold"], i["fold"] = r, r, k, k
        cv_preds.append(p)
        cv_info.append(i)
    print(f"  repeat {r + 1}/{N_REPEATS} done [{time.time() - t0:.0f}s]")
cv_preds = pd.concat(cv_preds, ignore_index=True)
cv_info = pd.concat(cv_info, ignore_index=True)
cv_info.to_csv(OUT / "prediction_split_info_cv.csv", index=False)

# per-fold metrics -> distribution of differences
fold_rows = []
for (r, k), g in cv_preds.groupby(["repeat", "fold"]):
    m = metric_table(g)
    m["repeat"], m["fold"] = r, k
    fold_rows.append(m)
fold_m = pd.concat(fold_rows)
wide = fold_m.pivot_table(index=["repeat", "fold", "category"], columns="method",
                          values=["log_loss", "accuracy", "brier"])
cv_diff = []
for c in list(CATEGORIES) + ["macro_average", "pooled_rows"]:
    w = wide.xs(c, level="category")
    for m in ["category", "shrunk"]:
        for metric in ["log_loss", "brier", "accuracy"]:
            d = w[(metric, m)] - w[(metric, "overall")]
            # mean over repeats of the per-repeat average (each repeat uses all rows once)
            per_rep = d.groupby(level="repeat").mean()
            cv_diff.append({"category": c, "method": m, "metric": metric,
                            "mean_diff": d.mean(), "sd_across_folds": d.std(),
                            "min_rep_mean": per_rep.min(), "max_rep_mean": per_rep.max(),
                            "share_folds_better": float((d < 0).mean() if metric != "accuracy" else (d > 0).mean())})
cv_diff = pd.DataFrame(cv_diff)
# pooled out-of-fold metrics per repeat, averaged
cv_metrics = (pd.concat([metric_table(g).assign(repeat=r) for r, g in cv_preds.groupby("repeat")])
              .groupby(["category", "method"], sort=False)[["log_loss", "accuracy", "brier", "n"]].mean()
              .reset_index())
print(cv_metrics.round(4).to_string())
print(cv_diff.round(4).to_string())
# session-bootstrap CI on cross-fitted predictions (repeat 0: every row held out once)
diff_cv_boot = paired_bootstrap(cv_preds[cv_preds.repeat == 0], np.random.default_rng(SEED + 2))
print("paired session-bootstrap CIs on cross-fitted (out-of-fold) predictions, repeat 0:")
print(diff_cv_boot.round(4).to_string())
diff_cv_boot.to_csv(OUT / "prediction_diff_cv_bootstrap.csv", index=False)
# effect size: share of the overall model's improvement over a coin flip
eff = cv_metrics.pivot(index="category", columns="method", values="log_loss")
eff["overall_gain_vs_coinflip"] = np.log(2) - eff["overall"]
for m in ["category", "shrunk"]:
    eff[f"{m}_extra_gain_pct_of_overall_gain"] = 100 * (eff["overall"] - eff[m]) / eff["overall_gain_vs_coinflip"]
print(eff.round(4).to_string())
eff.to_csv(OUT / "prediction_effect_size.csv")
cv_metrics.to_csv(OUT / "prediction_metrics_cv.csv", index=False)
cv_diff.to_csv(OUT / "prediction_diff_cv.csv", index=False)
print("lambda chosen by inner CV (counts):")
print(cv_info.groupby("category")["lambda"].value_counts().unstack(fill_value=0).to_string())

# train vs test (generalisation gap), primary + CV average
tt = []
for c in CATEGORIES:
    ci = cv_info[cv_info.category == c]
    for m in METHODS:
        tt.append({"category": c, "method": m,
                   "train_logloss_cv_mean": ci[f"train_logloss_{m}"].mean(),
                   "test_logloss_cv_mean": cv_metrics.query("category == @c and method == @m").log_loss.iloc[0]})
tt = pd.DataFrame(tt)
tt["gap"] = tt.test_logloss_cv_mean - tt.train_logloss_cv_mean
print(tt.round(4).to_string())
tt.to_csv(OUT / "prediction_train_vs_test.csv", index=False)

# =====================================================================
# 3. RECOMMENDATION EVALUATION (cross-fitted, repeat 0: every row once)
# =====================================================================
section("RECOMMENDATIONS (cross-fitted, repeat 0)")
rep0 = cv_preds[cv_preds.repeat == 0]
info0 = cv_info[cv_info.repeat == 0]
stab = []
for c in CATEGORIES:
    ci = info0[info0.category == c]
    stab.append({"category": c,
                 "rec_overall_by_fold": "; ".join(ci.rec_overall),
                 "rec_category_by_fold": "; ".join(ci.rec_category),
                 "rec_category_raw_by_fold": "; ".join(ci.rec_category_raw),
                 "rec_shrunk_by_fold": "; ".join(ci.rec_shrunk),
                 "folds_where_category_rec_differs": int((ci.rec_category != ci.rec_overall).sum()),
                 "folds_where_shrunk_rec_differs": int((ci.rec_shrunk != ci.rec_overall).sum())})
stab = pd.DataFrame(stab)
print(stab.T.to_string())
stab.to_csv(OUT / "recommendation_stability.csv", index=False)

# (a) direct battles + (c) held-out judge
battles, judge = [], []
for c in CATEGORIES:
    for k in range(N_OUTER):
        ci = info0[(info0.category == c) & (info0.fold == k)].iloc[0]
        held = rep0[(rep0.category == c) & (rep0.fold == k)]
        for which in ["category", "shrunk"]:
            r_o, r_x = ci.rec_overall, ci[f"rec_{which}"]
            if r_o == r_x:
                judge.append({"category": c, "fold": k, "vs": which, "rec_overall": r_o,
                              "rec_other": r_x, "same": True, "diff": 0.0, "se": 0.0})
                continue
            hb = held[((held.model_a == r_o) & (held.model_b == r_x)) |
                      ((held.model_a == r_x) & (held.model_b == r_o))]
            wins_x = int(((hb.model_a == r_x) & (hb.y == 1)).sum() + ((hb.model_b == r_x) & (hb.y == 0)).sum())
            battles.append({"category": c, "fold": k, "vs": which, "rec_overall": r_o,
                            "rec_other": r_x, "n_direct": len(hb), "wins_rec_other": wins_x})
            hd = dec.loc[held.row]
            ms = sorted(btl.model_counts(hd).index)
            if r_o in ms and r_x in ms:
                _, _, ns = btl.comparison_graph(hd, ms)
                fj = btl.fit_btl(hd, ms, l2=0.0 if ns == 1 else 0.01)
                d, s = btl.diff_se(fj, r_x, r_o)
                judge.append({"category": c, "fold": k, "vs": which, "rec_overall": r_o,
                              "rec_other": r_x, "same": False, "diff": d, "se": s,
                              "judge_strongly_connected": ns == 1})
battles = pd.DataFrame(battles)
judge = pd.DataFrame(judge)
battles.to_csv(OUT / "recommendation_direct_battles.csv", index=False)
judge.to_csv(OUT / "recommendation_heldout_judge_by_fold.csv", index=False)
jsum = (judge.groupby(["category", "vs"])
        .apply(lambda g: pd.Series({
            "folds_different": int((~g.same).sum()),
            "mean_diff_theta": g["diff"].mean(),
            "se_mean": np.sqrt((g.se ** 2).sum()) / len(g),
        }), include_groups=False).reset_index())
jsum["z"] = jsum.mean_diff_theta / jsum.se_mean.replace(0, np.nan)
jsum["implied_p_rec_other_beats_overall"] = expit(jsum.mean_diff_theta)
if len(battles):
    bsum = battles.groupby(["category", "vs"])[["n_direct", "wins_rec_other"]].sum().reset_index()
    bsum["win_rate_rec_other"] = bsum.wins_rec_other / bsum.n_direct.replace(0, np.nan)
    jsum = jsum.merge(bsum, on=["category", "vs"], how="left")
print(jsum.round(3).to_string())
jsum.to_csv(OUT / "recommendation_heldout_judge_summary.csv", index=False)

dis_cv = disagreement(rep0)
print(dis_cv.round(3).to_string())
dis_cv.to_csv(OUT / "recommendation_disagreement_cv.csv", index=False)

# =====================================================================
# 4. FIGURES
# =====================================================================
# calibration (pooled out-of-fold predictions, repeat 0 — every row once)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
bins = np.linspace(0, 1, 11)
for ax, (c, name) in zip(axes, CATEGORIES.items()):
    g = rep0[rep0.category == c]
    ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=1)
    for m, col, mk in [("overall", COLORS["Overall"], "o"), ("category_pred", COLORS[name], "s"),
                       ("shrunk", COLORS["Shrunk"], "^")]:
        b = np.clip(np.digitize(g[m], bins) - 1, 0, 9)
        dfb = pd.DataFrame({"p": g[m], "y": g.y, "b": b}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        dfb = dfb[dfb.n >= 20]
        label = {"overall": "Overall", "category_pred": "Category-only", "shrunk": "Shrunk"}[m]
        ax.plot(dfb.p, dfb.y, marker=mk, ms=5, lw=1.5, color=col, label=label)
    ax.set_title(f"{name} (n = {len(g):,})")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("predicted P(model A wins)")
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.1, 0.9)
axes[0].set_ylabel("observed share model A won")
fig.suptitle("Calibration of out-of-fold predictions (grouped 5-fold CV; bins with ≥ 20 rows)",
             fontsize=10, color=INK2)
fig.tight_layout()
fig.savefig(FIG / "calibration.png")
plt.close(fig)

# log-loss differences: primary bootstrap CI + CV fold distribution
fig, ax = plt.subplots(figsize=(8.5, 4))
cats_plot = list(CATEGORIES) + ["pooled_rows"]
lab = [CATEGORIES.get(c, "All category rows") for c in cats_plot]
for j, m in enumerate(["category", "shrunk"]):
    col = COLORS["Math"] if m == "category" else COLORS["Shrunk"]
    s = diff_p[(diff_p.metric == "log_loss") & (diff_p.method == m)].set_index("category").loc[cats_plot]
    yy = np.arange(len(cats_plot)) + (-0.15 if j == 0 else 0.15)
    ax.errorbar(s.diff_vs_overall, yy, xerr=[s.diff_vs_overall - s.ci_low, s.ci_high - s.diff_vs_overall],
                fmt="o", color=col, ms=6, capsize=3, label=f"{MLABEL[m]} − Overall (primary split, 95% CI)")
    cvd = fold_m[fold_m.method.isin([m, "overall"])].pivot_table(
        index=["repeat", "fold", "category"], columns="method", values="log_loss")
    cvd = (cvd[m] - cvd["overall"]).reset_index()
    for i, c in enumerate(cats_plot):
        v = cvd[cvd.category == c][0]
        ax.scatter(v, np.full(len(v), yy[i]) + np.random.default_rng(i).uniform(-0.06, 0.06, len(v)),
                   s=6, color=col, alpha=0.3, lw=0)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(range(len(cats_plot)), lab)
ax.invert_yaxis()
ax.set_xlabel("difference in held-out log loss vs overall BTL  (← task-specific better | worse →)")
ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
ax.text(0.99, 0.02, "faint dots: 25 CV folds", transform=ax.transAxes, ha="right", fontsize=7, color=INK2)
fig.savefig(FIG / "prediction_diff.png")
plt.close(fig)
print(f"\nDone in {time.time() - t0:.0f}s")
