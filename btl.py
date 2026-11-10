"""
btl.py — reusable Bradley–Terry–Luce utilities (MODELING layer).

Model:  P(model_a beats model_b) = sigmoid(theta_a - theta_b + gamma + offset)

    theta   : latent strength of each LLM (log-odds scale)
    gamma   : presentation-position effect. In this dataset model_a is always
              shown in position A, so gamma > 0 means position A is favoured
              and gamma < 0 means position B is favoured. gamma = 0 is plain BTL.
    offset  : optional known per-row term (used for the "shrunk" category model,
              where offset = theta_overall[a] - theta_overall[b] and theta is
              then a category-specific *deviation* from the overall strength).

Identifiability. Adding the same constant to every theta leaves every
probability unchanged, so theta is only identified up to a constant. We impose
sum(theta) = 0 by writing theta = A @ beta, where A is a K x (K-1) orthonormal
basis of the subspace orthogonal to the all-ones vector. The likelihood is then
strictly concave in (beta, gamma) whenever the MLE exists, and the covariance
of the *centered* theta is A Cov(beta) A^T — no pseudo-inverse of a singular
matrix is needed.

Existence of the MLE (Zermelo 1929; Ford 1957). With decisive outcomes and no
penalty, finite estimates exist iff the directed "who beat whom" graph is
strongly connected. If some model never lost (or never won) against a subset,
its theta diverges to +/- infinity. `comparison_graph` checks this, and
`fit_btl` reports it so that sparse category fits are not silently trusted.

Changes relative to the original version (see README / report, Part A):
  * Newton–Raphson with step halving instead of L-BFGS on a near-singular
    problem; convergence is checked on the max |gradient|.
  * Covariance computed in the identified (K-1)-dimensional parametrisation
    (the original inverted H + 1e-8*I with pinv, which put ~1e8 variance on
    the unidentified direction and relied on catastrophic cancellation).
  * Optional session-clustered ("sandwich") standard errors.
  * Optional per-row weights, fractional outcomes (ties = 0.5) and offsets.
  * Explicit, documented fallback for models absent from the fit
    (predict_proba(..., fallback=...)) instead of silently using theta = 0.
  * Cluster bootstrap for rank uncertainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.special import expit

ELO_SCALE = 400 / np.log(10)  # converts natural-log-odds units into Elo points


# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------
def prepare_decisive(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only decisive outcomes and drop self-matches.

    Returns a copy with y = 1 if model_a won, 0 if model_b won.
    """
    out = df[df["model_a"] != df["model_b"]]
    out = out[out["winner"].isin(["model_a", "model_b"])].copy()
    out["y"] = (out["winner"] == "model_a").astype(float)
    return out


def prepare_with_ties(df: pd.DataFrame, include_both_bad: bool = True) -> pd.DataFrame:
    """Sensitivity-analysis data: ties (and optionally both_bad) scored y = 0.5.

    A tie contributes 0.5*log p + 0.5*log(1-p) to the log-likelihood, which is
    the convention used by LMArena's own leaderboard (a tie = half a win).
    """
    keep = ["model_a", "model_b", "tie"] + (["both_bad"] if include_both_bad else [])
    out = df[(df["model_a"] != df["model_b"]) & df["winner"].isin(keep)].copy()
    out["y"] = out["winner"].map({"model_a": 1.0, "model_b": 0.0,
                                  "tie": 0.5, "both_bad": 0.5})
    return out


def model_counts(df: pd.DataFrame) -> pd.Series:
    """Number of comparisons each model appears in."""
    return pd.concat([df["model_a"], df["model_b"]]).value_counts()


def comparison_graph(df: pd.DataFrame, models: list[str]):
    """Connectivity diagnostics of the comparison graph.

    Returns (n_components_undirected, labels_undirected, n_strong_components).
    Undirected connectivity is necessary for all thetas to be on one scale.
    Strong connectivity of the directed win graph (edge loser -> winner) is
    necessary and sufficient for the unpenalised MLE to exist.
    """
    n_comp, labels, n_strong, _ = graph_diagnostics(df, models)
    return n_comp, labels, n_strong


def graph_diagnostics(df: pd.DataFrame, models: list[str]):
    """Like comparison_graph, but also returns strong-component labels."""
    idx = {m: k for k, m in enumerate(models)}
    a = df["model_a"].map(idx).to_numpy()
    b = df["model_b"].map(idx).to_numpy()
    K = len(models)
    undirected = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(K, K))
    n_comp, labels = connected_components(undirected, directed=False)

    y = df["y"].to_numpy()
    dec = (y == 0) | (y == 1)          # only decisive rows define win edges
    winner = np.where(y[dec] == 1, a[dec], b[dec])
    loser = np.where(y[dec] == 1, b[dec], a[dec])
    directed = sparse.coo_matrix((np.ones(dec.sum()), (loser, winner)), shape=(K, K))
    n_strong, s_labels = connected_components(directed, directed=True, connection="strong")
    return n_comp, labels, n_strong, s_labels


def wins_losses(df: pd.DataFrame) -> pd.DataFrame:
    """Decisive wins and losses per model (used to flag unbeaten/winless models)."""
    w = pd.concat([df.loc[df.y == 1, "model_a"], df.loc[df.y == 0, "model_b"]]).value_counts()
    l = pd.concat([df.loc[df.y == 0, "model_a"], df.loc[df.y == 1, "model_b"]]).value_counts()
    out = pd.DataFrame({"wins": w, "losses": l}).fillna(0).astype(int)
    return out


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------
def _design(df: pd.DataFrame, models: list[str]):
    idx = {m: k for k, m in enumerate(models)}
    a = df["model_a"].map(idx)
    b = df["model_b"].map(idx)
    if a.isna().any() or b.isna().any():
        raise ValueError("df contains models not in `models`")
    a, b = a.to_numpy(int), b.to_numpy(int)
    n, K = len(df), len(models)
    rows = np.r_[np.arange(n), np.arange(n)]
    cols = np.r_[a, b]
    vals = np.r_[np.ones(n), -np.ones(n)]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, K))


def _contrast_basis(K: int) -> np.ndarray:
    """K x (K-1) orthonormal basis of {theta : sum(theta) = 0}."""
    # Helmert-type construction via QR of [1, I]
    M = np.eye(K)
    M[:, 0] = 1.0
    Q, _ = np.linalg.qr(M)
    return Q[:, 1:]


def fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                 offset: np.ndarray | None = None, pen: np.ndarray | None = None,
                 cluster_codes: np.ndarray | None = None,
                 max_iter: int = 100, tol: float = 1e-9) -> dict:
    """Generic (ridge-penalised, weighted) logistic regression by Newton–Raphson.

    Minimises  sum_i w_i [log(1+e^{z_i}) - y_i z_i] + 0.5 * sum_k pen_k par_k^2,
    z = X @ par + offset, with step-halving line search. y may be fractional.
    Returns parameters, fitted probabilities, model-based covariance
    (inverse penalised Hessian) and, if cluster_codes is given, the
    cluster-robust sandwich covariance with a G/(G-1) small-sample factor.
    """
    n, p = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    pen = np.zeros(p) if pen is None else np.asarray(pen, float)

    def objective(par):
        z = X @ par + off
        return np.sum(w * (np.logaddexp(0, z) - y * z)) + 0.5 * np.sum(pen * par ** 2)

    par = np.zeros(p)
    f = objective(par)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        z = X @ par + off
        pr = expit(z)
        g = X.T @ (w * (pr - y)) + pen * par
        if np.max(np.abs(g)) < tol * max(1.0, w.sum()) ** 0.5:
            converged = True
            break
        H = (X * (w * pr * (1 - pr))[:, None]).T @ X + np.diag(pen)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        t = 1.0
        while True:
            new = par - t * step
            fn = objective(new)
            if fn <= f + 1e-12 or t < 1e-10:
                break
            t *= 0.5
        par, f = new, fn

    z = X @ par + off
    pr = expit(z)
    H = (X * (w * pr * (1 - pr))[:, None]).T @ X + np.diag(pen)
    cond = np.linalg.cond(H)
    Hinv = np.linalg.inv(H) if np.isfinite(cond) and cond < 1e14 else np.linalg.pinv(H)
    cov_rob = None
    if cluster_codes is not None:
        scores = X * (w * (pr - y))[:, None]
        G = int(cluster_codes.max()) + 1
        S = np.zeros((G, p))
        np.add.at(S, cluster_codes, scores)
        meat = S.T @ S * G / max(G - 1, 1)
        cov_rob = Hinv @ meat @ Hinv
    loglik = float(np.sum(w * (y * np.log(np.clip(pr, 1e-15, 1)) +
                               (1 - y) * np.log(np.clip(1 - pr, 1e-15, 1)))))
    return {"par": par, "p": pr, "cov": Hinv, "cov_robust": cov_rob,
            "loglik": loglik, "converged": bool(converged), "n_iter": it,
            "condition": float(cond)}


def fit_btl(df: pd.DataFrame, models: list[str] | None = None,
            position_effect: bool = True, l2: float = 0.0,
            weights: np.ndarray | None = None, offset: np.ndarray | None = None,
            cluster: str | None = "evaluation_session_id",
            max_iter: int = 100, tol: float = 1e-9) -> dict:
    """Penalised maximum-likelihood BTL fit by Newton–Raphson.

    Parameters
    ----------
    df : rows with model_a, model_b and outcome y in [0, 1].
    models : model list defining theta's index (default: all models in df).
    position_effect : estimate gamma (else gamma = 0).
    l2 : ridge penalty 0.5*l2*||theta||^2 (never applied to gamma). l2 = 0 is
         plain MLE. With an offset, this shrinks theta (the deviation) to 0.
    weights : optional non-negative per-row weights.
    offset : optional per-row additive term on the logit scale.
    cluster : column of df to cluster on for robust (sandwich) SEs; None to skip.

    Returns a dict; `se` are model-based (inverse Fisher information) and
    `se_robust` are session-clustered sandwich SEs (if cluster is given).
    """
    if models is None:
        models = sorted(pd.unique(df[["model_a", "model_b"]].to_numpy().ravel()))
    models = list(models)
    K = len(models)
    Xt = _design(df, models)                       # n x K (theta part)
    A = _contrast_basis(K)                         # K x (K-1)
    Xb = Xt @ A                                    # n x (K-1) dense
    X = np.column_stack([Xb, np.ones(len(df))]) if position_effect else Xb
    p = X.shape[1]
    y = df["y"].to_numpy(float)
    w = np.ones(len(df)) if weights is None else np.asarray(weights, float)
    off = np.zeros(len(df)) if offset is None else np.asarray(offset, float)
    pen = np.zeros(p)
    pen[:K - 1] = l2                               # ||A beta||^2 = ||beta||^2

    codes = pd.factorize(df[cluster])[0] if (cluster is not None and cluster in df.columns) else None
    r = fit_logistic(X, y, w=w, offset=off, pen=pen, cluster_codes=codes,
                     max_iter=max_iter, tol=tol)
    par, pr, cond, converged, it = r["par"], r["p"], r["condition"], r["converged"], r["n_iter"]

    # Map from (beta, gamma) to (theta, gamma)
    T = np.zeros((K + (1 if position_effect else 0), p))
    T[:K, :K - 1] = A
    if position_effect:
        T[K, K - 1] = 1.0
    full = T @ par
    cov = T @ r["cov"] @ T.T
    cov_rob = None if r["cov_robust"] is None else T @ r["cov_robust"] @ T.T

    theta = pd.Series(full[:K], index=models)
    loglik = r["loglik"]
    out = {
        "models": models,
        "theta": theta,
        "se": pd.Series(np.sqrt(np.diag(cov)[:K]), index=models),
        "gamma": float(full[K]) if position_effect else 0.0,
        "gamma_se": float(np.sqrt(cov[K, K])) if position_effect else np.nan,
        "cov": cov,
        "loglik": loglik,
        "n": len(df),
        "converged": bool(converged),
        "n_iter": it,
        "max_abs_theta": float(np.max(np.abs(theta))),
        "hessian_condition": float(cond),
        "l2": l2,
        "position_effect": position_effect,
    }
    if cov_rob is not None:
        out["se_robust"] = pd.Series(np.sqrt(np.diag(cov_rob)[:K]), index=models)
        out["gamma_se_robust"] = float(np.sqrt(cov_rob[K, K])) if position_effect else np.nan
        out["cov_robust"] = cov_rob
    return out


def diff_se(fit: dict, i: str, j: str, robust: bool = True) -> tuple[float, float]:
    """Estimate and SE of theta_i - theta_j (uses the full covariance)."""
    cov = fit["cov_robust"] if (robust and "cov_robust" in fit) else fit["cov"]
    k = {m: n for n, m in enumerate(fit["models"])}
    a, b = k[i], k[j]
    d = fit["theta"][i] - fit["theta"][j]
    v = cov[a, a] + cov[b, b] - 2 * cov[a, b]
    return float(d), float(np.sqrt(max(v, 0)))


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
def predict_proba(fit: dict, df: pd.DataFrame, fallback: dict | None = None,
                  offset: np.ndarray | None = None) -> np.ndarray:
    """P(model_a wins) for each row.

    Models absent from `fit` are handled explicitly:
      * fallback=None  -> raise an error (no silent theta = 0);
      * fallback=dict with key "theta" (a Series) and "gamma" -> rows where
        either model is missing from `fit` are predicted entirely by the
        fallback fit (e.g. the overall model). This keeps each prediction
        internally on one scale rather than mixing thetas from two fits.
    """
    th = fit["theta"]
    known = df["model_a"].isin(th.index) & df["model_b"].isin(th.index)
    off = np.zeros(len(df)) if offset is None else np.asarray(offset, float)
    p = np.full(len(df), np.nan)
    kk = known.to_numpy()
    if kk.any():
        p[kk] = expit(df.loc[known, "model_a"].map(th).to_numpy()
                      - df.loc[known, "model_b"].map(th).to_numpy()
                      + fit["gamma"] + off[kk])
    if (~kk).any():
        if fallback is None:
            missing = set(df.loc[~known, "model_a"]) | set(df.loc[~known, "model_b"])
            raise ValueError(f"models not in fit and no fallback: {sorted(missing - set(th.index))}")
        p[~kk] = predict_proba(fallback, df.loc[~known])
    return p


# ----------------------------------------------------------------------
# Tables and uncertainty
# ----------------------------------------------------------------------
def ranking_table(fit: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Tidy table: theta, SEs, 95% CIs, Elo-scale score, n, raw win rate.

    The CI is for theta relative to the average model (sum-to-zero), not an
    absolute quantity. Use diff_se() for pairwise comparisons.
    """
    th, se = fit["theta"], fit["se"]
    n = model_counts(df)
    wl = wins_losses(df).reindex(th.index).fillna(0)
    t = pd.DataFrame({
        "theta": th,
        "se": se,
        "ci_low": th - 1.96 * se,
        "ci_high": th + 1.96 * se,
        "elo": 1000 + ELO_SCALE * th,
        "n_comparisons": n.reindex(th.index).fillna(0).astype(int),
        "raw_win_rate": wl["wins"] / n.reindex(th.index),
    })
    if "se_robust" in fit:
        t.insert(2, "se_robust", fit["se_robust"])
        t["ci_low_robust"] = th - 1.96 * fit["se_robust"]
        t["ci_high_robust"] = th + 1.96 * fit["se_robust"]
    t = t.sort_values("theta", ascending=False)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    return t


def bootstrap_ranks(df: pd.DataFrame, models: list[str], B: int = 200,
                    seed: int = 0, cluster: str = "evaluation_session_id",
                    position_effect: bool = True, l2: float = 0.0) -> dict:
    """Session-cluster bootstrap of BTL ranks.

    Resamples evaluation sessions with replacement (keeping all comparisons
    from a session together, which respects within-session correlation),
    refits, and records each model's rank. Models absent from a bootstrap
    sample get rank NaN for that replicate.
    Returns rank matrix (B x K), theta matrix, and number of replicates in
    which the resampled win graph was not strongly connected.
    """
    # Drawing G sessions with replacement is equivalent to giving every
    # session a Multinomial(G, 1/G) count and weighting its rows by it.
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(df[cluster])
    G = len(uniq)
    ranks = np.full((B, len(models)), np.nan)
    thetas = np.full((B, len(models)), np.nan)
    not_strong = 0
    for b in range(B):
        cnt = rng.multinomial(G, np.full(G, 1.0 / G))
        wrow = cnt[codes].astype(float)
        keep = wrow > 0
        d = df.loc[keep]
        present = [m for m in models if m in set(d.model_a) | set(d.model_b)]
        _, _, ns = comparison_graph(d, present)
        not_strong += int(ns > 1)
        f = fit_btl(d, present, position_effect=position_effect,
                    l2=l2, cluster=None, weights=wrow[keep])
        r = f["theta"].rank(ascending=False, method="min")
        ix = [models.index(m) for m in present]
        ranks[b, ix] = r.to_numpy()
        thetas[b, ix] = f["theta"].to_numpy()
    return {"ranks": pd.DataFrame(ranks, columns=models),
            "thetas": pd.DataFrame(thetas, columns=models),
            "not_strongly_connected": not_strong, "B": B}


def rank_summary(boot: dict) -> pd.DataFrame:
    """Per-model 95% bootstrap rank interval and P(rank 1), P(top 3)."""
    R = boot["ranks"]
    return pd.DataFrame({
        "rank_ci_low": R.quantile(0.025).round().astype("Int64"),
        "rank_ci_high": R.quantile(0.975).round().astype("Int64"),
        "p_rank1": (R == 1).mean(),
        "p_top3": (R <= 3).mean(),
    })


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def log_loss_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"log_loss": float(np.mean(log_loss_rows(y, p))),
            "accuracy": float(np.mean((p > 0.5) == (y == 1))),
            "brier": float(np.mean((p - y) ** 2)),
            "n": int(len(y))}
