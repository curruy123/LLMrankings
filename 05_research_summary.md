# Task-specific rankings of LLMs from human pairwise preferences

### An exploratory Bradley–Terry–Luce analysis of `lmarena-ai/arena-human-preference-140k`

*Exploratory research project suggested by Professor Ying Jin (UPenn). All numbers below were produced by the scripts in this folder (`run_all.sh`) and can be traced to the CSV files and figures named in each section.*

---

## Summary: three separate findings

The three questions have **different answers**, so it matters not to blur them.

| Question | Answer | Strength of evidence |
|---|---|---|
| **1. Do rankings vary across tasks?** | **Yes.** In all three categories, a likelihood-ratio test rejects "same ranking, up to a uniform rescaling" (math p ≈ 2×10⁻¹⁴, coding p ≈ 5×10⁻²⁰, creative writing p ≈ 2×10⁻⁴). For math and coding, rank agreement with the overall ranking is lower than sampling noise alone would produce. Several models are identifiable task specialists (e.g. minimax-m1 in math; Claude 4 models in coding). | Strong for math and coding; moderate for creative writing. |
| **2. Does task-specific BTL predict held-out preferences better?** | **Yes, but the gain is small, and it depends on how the model is built.** A partially pooled model (overall strength plus a penalised task deviation) beats the overall model in every category and in all 25 CV folds: pooled held-out log loss falls from 0.6591 to 0.6557, which is ≈10% of the overall model's own gain over a coin flip. A *pure* category-only model helps in math and coding but **does not help in creative writing** (log loss +0.0007, CI −0.004 to +0.006), and it overfits noticeably. | Strong (CIs exclude zero) for the pooled model; mixed for category-only. |
| **3. Does the evidence support more reliable task-specific *recommendations*?** | **Only partly.** For choosing between two given models, task-aware rankings agree with held-out human votes more often where they disagree with the overall ranking (math: 57% vs 43%, n = 474). For the **top-1 recommendation**, the data do **not** reliably support replacing the overall #1 (Gemini 2.5 Pro). The task #1 differs only in math (minimax-m1), and that difference is not statistically significant (held-out estimated win probability 0.57, z = 1.1). In creative writing, the category-only #1 was *worse* on held-out data (z = −2.4). | Weak/inconclusive for top-1 recommendations; moderate for pairwise choices. |

**In one sentence:** rankings genuinely shift with the task, and modelling that shift slightly improves prediction. With this much data, though, the shift is mostly visible in the middle of the leaderboard; at the top, the evidence is too thin to say that a task-specific "best model" is better than the overall best.

---

## 1. Research question and motivation

Public leaderboards like LMArena summarise each LLM by a single strength score. Users, however, care about specific tasks. Professor Jin's suggestion was to fit a Bradley–Terry–Luce (BTL) model to the pairwise human preferences, examine how rankings differ across task categories, and test whether task-specific rankings predict human preferences better than an overall ranking. On top of that, we ask whether task-specific rankings lead to **more reliable recommendations** — a related but distinct question, because:

- a ranking can change **without** the change being distinguishable from noise;
- a model can have better average **log loss** while its **top-ranked** model is no better (log loss is averaged over all pairs, mostly mid-table ones, while a recommendation depends only on the top).

## 2. Data and preprocessing

| Step | Rows | Source |
|---|---|---|
| Full dataset | 135,634 votes, 53 LLMs, 126 languages | `research_results.txt`, `log_01_overall.txt` |
| English only (`language == "en"`) | 71,175 | |
| Drop 10 self-matches, 11,376 ties, 8,135 `both_bad` | **51,659 decisive comparisons, 52 LLMs** (27.4% of English rows dropped) | |
| Sessions | 45,821 evaluation sessions; 8.7% contain >1 vote (19% of rows) | |
| Pair coverage | 1,201 of 1,326 possible model pairs observed | |

`qwen3-coder-480b-a35b-instruct` appears only in non-English data, which is why the English analysis has 52 rather than 53 models.

**Category indicators** come from the cleaned CSV: `math` (from `math_v0.1`), `coding` (`is_code`), `creative_writing` (`creative_writing_v0.1`). **The categories overlap** (`outputs/category_overlap.csv`):

| English decisive | Math | Coding | Creative writing |
|---|---|---|---|
| Math | **3,833** | 1,117 | 22 |
| Coding | 1,117 | **17,522** | 320 |
| Creative writing | 22 | 320 | **4,248** |

29.6% of math rows are also coding rows, so math and coding are *not* independent samples. 27,510 rows (53%) belong to no category. We use every row that carries the flag (overlapping rows enter several fits) and test exclusive-row versions in the sensitivity analysis (§8, S4).

**Verification of the preliminary figures.** All the numbers you reported reproduce exactly: 135,634 evaluations, 53 models, 51,659 decisive English comparisons with 52 models, Gemini 2.5 Pro ranked #1, and Spearman(BTL, raw win rate) = 0.9965. The re-run of the original `01_fit_overall.py` matched the uploaded `overall_ranking.csv` to 2×10⁻¹³.

## 3. BTL formulation

For a comparison between model *a* (shown in position A) and model *b* (position B):

$$
P(a \succ b) = \sigma(\theta_a - \theta_b + \gamma), \qquad \sigma(x) = \frac{1}{1+e^{-x}}
$$

- **θᵢ** is a latent strength on the log-odds scale. A gap of 0.4 means the stronger model is preferred about 60% of the time. Only differences are identified, so we impose Σθᵢ = 0 (θ is "strength relative to the average model"). Elo-like scores are 1000 + (400/ln 10)·θ.
- **γ** is a presentation-position effect (γ > 0 favours position A).
- **Estimation:** maximum likelihood by Newton–Raphson in a (K−1)-dimensional contrast parametrisation (θ = Aβ, with A an orthonormal basis orthogonal to the ones vector). This keeps the Hessian non-singular and gives the covariance of the centred θ directly.
- **Existence:** a finite MLE exists if and only if the directed "who-beat-whom" graph is strongly connected (Zermelo 1929; Ford 1957). We check this for every fit.
- **Uncertainty:** (i) model-based SEs (inverse Fisher information); (ii) session-clustered sandwich SEs; (iii) a session-cluster bootstrap (200 refits) that gives a 95% interval for each model's **rank** and P(rank = 1).

**Why BTL here?** It is the natural model for paired comparisons, it adjusts for opponent strength (a model that mostly meets strong opponents is not penalised), and it gives a probability for every possible matchup, which is what held-out evaluation needs.

### Part A — audit of the original implementation

| Check | Finding | Action |
|---|---|---|
| Likelihood and gradient | Correct (numerical gradient check error 1×10⁻⁴). | Kept. |
| Identifiability / centring | Handled by a 1e-8 ridge, then centring. The covariance came from `pinv(H + 1e-8·I)`, which puts ~10⁸ variance on the unidentified direction and relies on cancellation when centring. It gave correct SEs here only by luck of conditioning. | Replaced with an exact contrast parametrisation (Hessian condition number 303). |
| Convergence | L-BFGS stopped slightly early (max \|gradient\| 0.17; log-likelihood 0.0006 below the optimum; θ off by ≤ 0.004). | Newton–Raphson converges in 5 iterations to gradient ≈ 0. Ranking unchanged. |
| Graph connectivity | Already checked; overall graph is connected and strongly connected. | Added strong-component labels and zero-win/zero-loss flags (essential for sparse categories). |
| SEs and CIs | Model-based only; CIs are per model, so they cannot tell you whether two models differ. | Added session-robust SEs, pairwise-difference SEs (`btl.diff_se`) and bootstrap rank intervals. |
| Position effect | γ = −0.0205 (SE 0.009, p = 0.024): position **B** is slightly favoured (A wins 49.5% of decisive votes between equal models). | Interpretation stated; removing γ leaves all rankings unchanged (Spearman 1.000). |
| Ties / both_bad | Dropped silently (27.4% of English data). Tie rate correlates *positively* with strength (ρ = 0.54) and `both_bad` *negatively* (ρ = −0.71), so exclusion is not obviously innocuous. | Sensitivity analysis (S1). |
| Low-sample models | Flag at < 50 comparisons only; no protection against non-existent MLEs. | Explicit reliability flags in category fits (§5). |
| Unseen models in prediction | `predict_proba` silently assigned θ = 0. | Now raises an error unless an explicit fallback is given (§6). |

## 4. Overall ranking

`outputs/overall_ranking.csv`, `log_01_overall.txt`

| Rank | Model | θ (robust SE) | Elo | 95% rank interval | P(rank 1) |
|---|---|---|---|---|---|
| 1 | gemini-2.5-pro | 0.865 (0.038) | 1150 | 1–2 | 0.745 |
| 2 | gemini-2.5-pro-preview-03-25 | 0.792 (0.083) | 1138 | 1–3 | 0.235 |
| 3 | grok-4-0709 | 0.662 (0.087) | 1115 | 2–7 | 0.020 |
| 4 | deepseek-r1-0528 | 0.557 (0.043) | 1097 | 3–8 | 0 |
| 5 | o3-2025-04-16 | 0.556 (0.036) | 1097 | 3–8 | 0 |

- **Interpretation.** Gemini 2.5 Pro would be preferred over an average model about 70% of the time (σ(0.865) = 0.70). Its lead over Gemini 2.5 Pro preview-03-25 is **not significant** (difference 0.073, SE 0.092, z = 0.79; implied head-to-head preference 51.8%), and it is #1 in only 74.5% of bootstrap refits.
- **BTL vs raw win rate:** Spearman 0.9965. Because Arena pairs models fairly evenly, opponent adjustment barely matters overall; the largest shift is 4 places (gemini-2.0-flash-thinking-exp, 166 comparisons).
- **Uncertainty is uneven:** the median rank-interval width is 6 places, and it is wider for models with few comparisons (Spearman between n and width = −0.40; `figures/rank_ci_vs_n.png`).

## 5. Task-specific rankings (Part B)

`outputs/{math,coding,creative_writing}_ranking.csv`, `category_fit_summary.csv`, `category_top_models.csv`

All three category fits converged; each comparison graph is connected and strongly connected, so the MLE exists for every model. Every model has at least one win and one loss in every category. **Reliability flag** (stated rule, not silent): a model is flagged if it has < 50 category comparisons **or** a robust SE > 0.25 logits (a 95% CI wider than about ±12 percentage points of win probability against an equal opponent).

| | Math | Coding | Creative writing |
|---|---|---|---|
| Comparisons / models / sessions | 3,833 / 52 / 3,540 | 17,522 / 52 / 15,581 | 4,248 / 52 / 3,766 |
| Comparisons per model (min / median) | 18 / 159 | 57 / 697 | 15 / 174 |
| Models flagged unreliable | **15** | 1 | 7 |
| Position effect γ (p) | +0.030 (0.40) | −0.030 (0.055) | −0.010 (0.76) |

**Top of each category, with uncertainty:**

| Category | #1 (P(rank 1), rank 95% CI) | Overall #1 Gemini 2.5 Pro in this category | #1 minus Gemini 2.5 Pro (θ, SE, p) | Other models with P(rank 1) ≥ 5% |
|---|---|---|---|---|
| Math | **minimax-m1** (0.69, 1–3) | rank 2, P(rank 1) = 0.10 | +0.219, 0.228, p = 0.34 | qwen3-235b-a22b, gemini-2.5-pro-preview-03-25*, grok-4-0709* |
| Coding | **gemini-2.5-pro** (0.56, 1–3) | rank 1 | — | gemini-2.5-pro-preview-03-25 (0.40) |
| Creative writing | **gemini-2.5-pro-preview-03-25*** (0.34, 1–7) | rank 3, P(rank 1) = 0.11 | +0.174, 0.332, p = 0.60 | gemini-2.0-flash-thinking-exp* (0.43 with only 17 comparisons!) |

\* = flagged as unreliable in that category.

**What this allows us to conclude.**

- The best-ranked model **differs from the overall best in math and creative writing**, but **neither difference is statistically significant**.
- In creative writing, the #1 is itself flagged (56 comparisons). The model with the highest bootstrap P(rank 1) has only 17 comparisons, which is a textbook small-sample artefact. Among reliably estimated models, the creative-writing #1 is Gemini 2.5 Pro.
- Minimax-m1's math #1 is the most credible task-specific top pick: it is reliably estimated (194 comparisons) and #1 in 69% of bootstrap samples. Even so, its advantage over Gemini 2.5 Pro is within noise.

## 6. Overall vs category rankings (Part C)

`outputs/ranking_comparison_summary.csv`, `rank_changes.csv`, `specialisation.csv`, `category_rank_matrix.csv`; figures `rank_scatter.png`, `rank_heatmap.png`, `largest_shifts.png`, `spearman_null.png`, `specialisation.png`

**Why not just compare θ across separate fits?** Each fit is centred on its own average model, and the spread of θ depends on how decisive voters are in that task. We therefore use (i) **ranks** (scale-free) and (ii) a **joint model** fitted to all English decisive data in a single likelihood, which estimates a separate strength vector for category rows and for the remaining rows (each with its own γ). Three nested versions:

- *same*: θ_c = θ_rest (task does not matter);
- *scale*: θ_c = κ·θ_rest (same ranking; votes more or less decisive, κ);
- *sep*: θ_c free (task-specific ranking).

**Test 1 — do strengths differ? Test 2 — do *rankings* differ beyond a uniform stretch?**

| | Math | Coding | Creative writing |
|---|---|---|---|
| κ (spread vs rest) [95% CI] | 1.06 [0.94, 1.18] | **0.86 [0.80, 0.92]** | 0.93 [0.83, 1.03] |
| LR *sep* vs *same* (51 df) | 167.1, p = 3×10⁻¹⁴ | 219.1, p = 1×10⁻²² | 95.2, p = 2×10⁻⁴ |
| Session-robust Wald, same test | 152.7, p = 4×10⁻¹² | 215.6, p = 4×10⁻²² | 93.5, p = 3×10⁻⁴ |
| LR *sep* vs *scale* (50 df) | 166.3, p = 2×10⁻¹⁴ | 201.4, p = 5×10⁻²⁰ | 93.9, p = 2×10⁻⁴ |

Coding votes are *less* decisive than other votes (κ < 1): strength gaps shrink by about 14% on coding prompts. In every category the ranking itself differs, not only the spread.

**Test 3 — is the observed rank disagreement larger than sampling noise?** We simulated 200 datasets in which the category votes, on the exact same matchups, are generated by the *overall* strengths (scaled by κ), then refitted. If tasks did not matter, this is how much ranks would move from noise alone.

| | Math | Coding | Creative writing |
|---|---|---|---|
| Spearman ρ with overall (all 52) | 0.824 | 0.871 | 0.856 |
| Null median [2.5% quantile] | 0.889 [0.833] | 0.956 [0.925] | 0.882 [0.812] |
| P(null ρ ≤ observed) | **0.030** | **0.005** | 0.219 |
| Spearman among reliable models (n) | 0.769 (37) | 0.884 (51) | 0.881 (45) |
| P(null ≤ observed), reliable | **0.005** | **0.005** | 0.075 |
| Mean \|rank change\|: observed / null | 6.7 / 5.1 | 5.1 / 3.1 | 6.5 / 5.2 |

A large share of the rank movement is pure noise: under the null, ranks still move by 3–5 places on average. Beyond that, math and coding show **more** disagreement than noise explains. For creative writing the evidence is borderline (rank-based test p = 0.22 for all models, 0.075 for reliable ones), even though the likelihood-ratio test is clearly significant. The LR test uses all the information in the strengths, while Spearman only uses ranks.

(Note: the bootstrap CI for Spearman in `ranking_comparison_summary.csv` is biased downward, because resampling adds a second layer of noise. The null simulation is the right reference.)

**Specialists** (`specialisation.csv`; δ = θ_c − κ·θ_rest, session-robust SE, Benjamini–Hochberg q < 0.10, reliable models only):

- **Math:** stronger: minimax-m1 (δ = +1.15, z = 6.2; overall #12 → math #1), qwen3-235b-a22b (+0.70; #18 → #3), o3-mini (+0.50; #39 → #23), claude-sonnet-4 (+0.48; #37 → #20). Weaker: deepseek-r1-0528 (−0.60; #4 → #28), gemma-3-27b-it (−0.60; #21 → #40).
- **Coding:** stronger: claude-sonnet-4-thinking-32k (+0.52; #31 → #12), claude-opus-4-thinking-16k (+0.36), claude-opus-4 (+0.28), gpt-4.1-mini (+0.29), mistral-small-2506 (+0.35), claude-sonnet-4 (+0.19). Weaker: minimax-m1 (−0.31; #12 → #30), gemma-3-27b-it, gemma-3n, llama-3.3-70b, amazon-nova-pro.
- **Creative writing:** only weaker: qwen3-30b-a3b (−0.63), gpt-4.1-mini (−0.49).

Hypothesis (not tested here): the coding pattern is consistent with the common perception that Claude 4 models are favoured for code. The deepseek-r1 drop in math could reflect long reasoning outputs being judged unfavourably by casual voters, but nothing in this dataset lets us check that.

## 7. Held-out prediction (Part D)

`outputs/prediction_*.csv`, `recommendation_*.csv`; figures `prediction_diff.png`, `calibration.png`

**Design.**

- **Splitting:** grouped by `evaluation_session_id`, so no session contributes to both training and test. The primary split is 80/20 with a fixed seed. Robustness comes from 5 repetitions of grouped 5-fold CV; in each repetition, every row is predicted exactly once from a model that never saw it.
- **Three predictors**, each fitted on training rows only and each with its own position effect:
  1. **Overall BTL** (all training comparisons).
  2. **Category-only BTL** (training rows in the category).
  3. **Shrunk / partial pooling:** logit = θ_overall,a − θ_overall,b + δ_a − δ_b + γ_c, with a ridge penalty λ‖δ‖². λ is chosen by an inner grouped CV on the training data only (chosen λ: math mostly 20, coding and creative writing mostly 50). λ → ∞ gives the overall model and λ → 0 gives category-only.
- **Fallback for models missing from category training data:** the whole row is predicted by the overall model (category-only), or δ = 0 (shrunk). In practice there were 0 fallback rows and 0 dropped rows: every model appeared in every category's training data.
- **Comparison:** all three predictors are scored on *exactly the same* test rows.

**Results (cross-fitted, all 25,603 category rows; paired session-bootstrap 95% CIs, repeat 0):**

| | Overall log loss | Category-only − overall | Shrunk − overall | Shrunk accuracy gain |
|---|---|---|---|---|
| Math (3,833) | 0.6512 | −0.0050 [−0.0119, +0.0021] | **−0.0091 [−0.0137, −0.0050]** | +1.9 pp [0.8, 3.0] |
| Coding (17,522) | 0.6605 | **−0.0023 [−0.0037, −0.0009]** | **−0.0025 [−0.0036, −0.0014]** | +0.5 pp [0.1, 0.9] |
| Creative writing (4,248) | 0.6605 | +0.0007 [−0.0037, +0.0055] | **−0.0028 [−0.0047, −0.0009]** | +0.9 pp [0.1, 1.7] |
| All category rows | 0.6591 | **−0.0022 [−0.0038, −0.0006]** | **−0.0035 [−0.0046, −0.0025]** | +0.75 pp [0.4, 1.1] |

- **Primary 80/20 split:** same direction, wider intervals. Pooled shrunk −0.0044 [−0.0064, −0.0022]; category-only −0.0035 [−0.0067, −0.0001]; creative-writing category-only +0.0010 [−0.010, +0.012] (`prediction_diff_primary.csv`).
- **Stability across folds:** the shrunk model beat overall in **100%** of the 25 CV folds (pooled), and in 100% / 100% / 84% for math / coding / creative writing. Category-only beat overall in 80% / 80% / 32%.
- **Effect size:** the overall model reduces log loss below a coin flip (ln 2 = 0.693) by 0.034. The shrunk model adds about 10% to that (23% in math, 7% in coding, 5% in creative writing). Human preferences are very noisy: even the best model is correct on only about 61–63% of decisive votes.
- **Training fit vs generalisation** (`prediction_train_vs_test.csv`): the train–test log-loss gap is 0.0015 for the overall model, but 0.018 (math) and 0.015 (creative writing) for category-only, and about half that for shrunk. Category-only models fit their training data much better than they predict. This is exactly why partial pooling wins.
- **Calibration:** all three predictors are well calibrated across the 0.25–0.75 range (`calibration.png`). The differences are in sharpness, not in systematic bias.

**Conclusion for Q2:** the hypothesis *"task-specific BTL predicts held-out human preferences better than overall BTL"* is **supported for a partially pooled task-specific model in all three categories**. For a **pure** category-only model, it is supported only for coding (and the pooled total); in math the CI includes zero, and in creative writing there is no improvement. The improvement is statistically clear but practically small.

## 8. Recommendations (Professor Jin's question)

Here a recommendation is made **from training data only** and judged **on held-out votes**. The cross-fitted repeat-0 design means every held-out vote is used once. The category recommendation is the category-only #1 among reliably estimated models (≥ 50 training comparisons, SE < 0.25).

**8.1 Stability of the recommendation across the 5 training folds** (`recommendation_stability.csv`)

| | Overall rec | Category-only rec | Shrunk rec |
|---|---|---|---|
| Math | gemini-2.5-pro (5/5) | minimax-m1 (5/5) | gemini-2.5-pro (5/5) |
| Coding | gemini-2.5-pro (5/5) | gemini-2.5-pro (3/5), preview-03-25 (2/5) | gemini-2.5-pro (4/5) |
| Creative writing | gemini-2.5-pro (5/5) | gemini-2.5-pro (3/5), grok-3-preview (2/5) | gemini-2.5-pro (5/5) |

Without the reliability filter, the creative-writing category #1 was gemini-2.0-flash-thinking-exp (17 comparisons in the full category data, fewer in each training fold) in one fold. This shows why unfiltered category maxima make poor recommendations.

**8.2 Are task-specific recommendations preferred by held-out humans?**

- **(a) Direct head-to-head battles** between the two recommendations in held-out data are **too rare to evaluate**. Math: 7 battles, minimax-m1 won 4. Creative writing: 2 battles, grok-3-preview won 0. Coding: 0. This is a direct consequence of incomplete matchup coverage.
- **(b) Held-out BTL "judge":** in each fold, a BTL model fitted only to the held-out votes estimates θ(category rec) − θ(overall rec), using all indirect comparisons; results are averaged over the 5 disjoint folds (`recommendation_heldout_judge_summary.csv`).

| | Category-only rec vs overall rec | Shrunk rec vs overall rec |
|---|---|---|
| Math | +0.26 (SE 0.24, z = 1.1) → est. P(cat. rec preferred) = 0.57 | identical in 5/5 folds |
| Coding | −0.14 (SE 0.10, z = −1.4) | −0.06 (SE 0.07) |
| Creative writing | **−0.30 (SE 0.13, z = −2.4)** → category rec *worse* | identical in 5/5 folds |

- **(c) Pairwise decisions** (`recommendation_disagreement_cv.csv`): consider held-out votes where the two rankings disagree about which of the two LLMs is better. Whose pick did the human choose?

| | Category-only vs overall | Shrunk vs overall |
|---|---|---|
| Math | 54.0% vs 46.0% (n = 658, SE 1.9 pp) | **57.4% vs 42.6% (n = 474, SE 2.3 pp)** |
| Coding | 52.4% (n = 2,071, SE 1.1 pp) | 52.2% (n = 1,633, SE 1.2 pp) |
| Creative writing | 49.6% (n = 680) — no gain | **55.3% (n = 300, SE 2.9 pp)** |
| All | 52.1% (n = 3,409, SE 0.9 pp) | **53.6% (n = 2,407, SE 1.0 pp)** |

**What this allows us to conclude:**

- **For "which of these two models should I use for task X?"**, task-aware (especially partially pooled) rankings make better choices than the overall ranking. The gain is clearest in math.
- **For "what single model should I use for task X?"**, the evidence does **not** support overriding the overall #1. In math, the category recommendation (minimax-m1) is estimated to be somewhat better on held-out data, but not significantly so. In coding and creative writing, the category-only recommendation, when it differs, does slightly or significantly worse. The shrunk model keeps Gemini 2.5 Pro on top in nearly every fold, which matches this evidence.
- **Better log loss does not imply a better top-1 recommendation:** category-only improves coding log loss, yet its coding top-1 changes did not help. Different rankings also do not imply better recommendations: creative writing's ranking differs significantly (LR test), yet its category #1 did worse.

**Limits of observational recommendation evaluation:**

1. Users choose prompts but not opponents; whether two models ever meet is decided by the platform's sampling, so direct comparisons of the recommended models are rare.
2. Indirect (BTL) comparisons assume the BTL model holds, which partly reuses the modelling assumption being evaluated.
3. Each category's held-out prompts reflect who uses Arena for that task. A user population with different prompts could rank models differently.
4. Category labels are automatic classifier tags, not ground-truth tasks.
5. Human Arena votes measure *preference*, not correctness; a model preferred for math may not be more accurate at math.

## 9. Statistical uncertainty: how to read the numbers

- **Model strength CIs vs rank intervals.** A narrow θ CI does not mean a certain rank: models in the crowded middle (θ within ~0.05 of each other) swap ranks easily. Use the bootstrap rank interval and P(rank 1) for ranking claims, and `btl.diff_se` for "is A better than B?".
- **Multiple comparisons.** With 52 models × 3 categories, several "significant" individual shifts are expected by chance; we therefore report BH q-values and rely on joint tests (LR/Wald) for the main claim that rankings differ.
- **Session clustering** inflates SEs by only 0–3% (robust/model SE ratio mean 1.001, max 1.03), because 91% of sessions contribute a single vote.
- **The null simulation** shows that ranks would move by 3–5 places on average from sampling noise alone. Individual rank changes smaller than this should not be interpreted.

## 10. Limitations and robustness (Part E)

`outputs/sensitivity_summary.csv`, `log_04b_sensitivity.txt`, figures `gof_pairs.png`, `rank_ci_vs_n.png`

| Limitation | Sensitivity analysis | Result |
|---|---|---|
| Ties & `both_bad` excluded | Score them as half-wins | Spearman with main ranking ≥ 0.996 everywhere. Creative-writing #1 switches to gemini-2.0-flash-thinking-exp (17 comparisons — unreliable). Overall, math and coding #1 unchanged. |
| Position bias | Refit without γ | Spearman = 1.000 in all scopes. γ is small (A wins 49.5% between equal models). |
| Unequal matchup frequencies | Weight each observed pair equally | Spearman 0.94–0.99; #1 changes in math (→ gemini-2.5-pro) and coding (→ preview-03-25). |
| Overlapping categories | Category-exclusive rows | Math Spearman 0.92 (minimax-m1 still #1). Coding and creative #1 switch between near-tied Gemini models / an unreliable model. |
| Session correlation | One random vote per session; robust SEs | Spearman ≥ 0.97; SEs inflated ≤ 3%. |
| English only | All languages (98,341 decisive) | Spearman 0.94–0.98; math #1 becomes gemini-2.5-pro. |
| Sparse models | Ridge (l2 = 1); partial pooling | Ridge: Spearman ≥ 0.999. Partial pooling: math #1 → gemini-2.5-pro (minimax-m1 #2), creative #1 → gemini-2.5-pro. |
| Unequal comparisons per model | Rank-interval width vs n | Strongly negative (math ρ = −0.60); median rank-interval width 6 (overall) vs 12–20 (categories). |
| One-dimensional strength | Pair-level goodness of fit (876 pairs with ≥ 20 votes) | X²/df = 1.10 (p = 0.02), 5.6% of pairs with \|z\| > 1.96 vs 5% expected. Mild overdispersion only; the larger departures are *across tasks* (§6), i.e. strength is approximately one-dimensional *within* a task mix but not across tasks. |
| Training fit vs generalisation | §7 | Category-only overfits (gap 0.015–0.018 in small categories); pooling fixes most of it. |

**The pattern across all sensitivity checks:** the bulk of each ranking is very robust (Spearman ≥ 0.93 under every alternative), but **the identity of the #1 model in each category is fragile**. That is the same conclusion as the bootstrap: the top 2–3 models are statistically tied.

Other limitations not analysed: model versions entered the arena at different times (time trends are confounded with model identity); category tags are automatic; the English filter relies on automatic language detection; and the data cover a fixed window, so the rankings describe these models at that time, not current versions.

## 11. Conclusions

1. **BTL works well as a summary of overall preferences.** It converges cleanly, its MLE exists for all 52 models, it fits pair-level data with only mild overdispersion, and it is nearly identical to raw win rates (ρ = 0.9965), because Arena's matchups are fairly balanced. Gemini 2.5 Pro is #1 (P = 0.75), statistically tied with its 03-25 preview.
2. **Rankings vary across tasks.** The variation is beyond a rescaling and beyond sampling noise for math and coding, and weaker but still detectable for creative writing. Clear specialists exist (minimax-m1, qwen3-235b in math; Claude 4 models in coding).
3. **Task-specific information improves held-out prediction a little**, provided it is combined with the overall ranking (partial pooling). Pure per-task models overfit and fail to help in creative writing.
4. **Task-specific *recommendations* are better supported for pairwise choices than for picking a single best model.** With the current data, the evidence that any category has a better top-1 model than the overall #1 is at best suggestive (math: minimax-m1), and in creative writing a naive category #1 did worse on held-out data.

## 12. Follow-up research questions

1. **Hierarchical / multi-task BTL:** a Bayesian model with θ_c = θ + δ_c and δ_c ~ N(0, τ²I), estimating τ per category. This is a principled version of our shrunk model, and its posterior would give P(rank 1) that propagates all uncertainty.
2. **Low-rank (multidimensional) preference models** (e.g. blade-chest or a 2-D latent factor): is there a "reasoning" vs "style" axis that explains the math/coding/creative differences?
3. **Finer task taxonomy:** use the other `category_tag` fields (complexity, domain knowledge, problem solving, instruction following) and test which dimensions drive rank changes.
4. **Ties as information:** a Davidson or Rao–Kupper model that treats ties as a separate outcome, given that tie rates are correlated with strength.
5. **Adaptive matchup design:** how many additional direct battles between the top-3 models in each category would be needed to separate them, and could active sampling get there more cheaply?
6. **Temporal stability:** do task-specific rankings remain stable as new model versions enter (requires the timestamp column)?

---

### Files referenced

| Part | Script | Key outputs |
|---|---|---|
| A | `btl.py`, `01_fit_overall.py` | `overall_ranking.csv`, `overall_fit_summary.csv`, `overall_bootstrap_ranks.csv` |
| B | `02_fit_categories.py` | `{math,coding,creative_writing}_ranking.csv`, `category_fit_summary.csv`, `category_top_models.csv`, `category_overlap.csv` |
| C | `03_compare_rankings.py` | `rank_changes.csv`, `ranking_comparison_summary.csv`, `specialisation.csv`, `category_rank_matrix.csv`; figures `rank_scatter`, `rank_heatmap`, `largest_shifts`, `spearman_null`, `specialisation` |
| D | `04_evaluate_predictions.py` | `prediction_metrics_{primary,cv}.csv`, `prediction_diff_{primary,cv,cv_bootstrap}.csv`, `prediction_train_vs_test.csv`, `prediction_effect_size.csv`, `recommendation_*.csv`; figures `prediction_diff`, `calibration` |
| E | `04b_sensitivity.py` | `sensitivity_summary.csv`, `sensitivity_tie_rates.csv`, `sensitivity_gof_pairs.csv`, `shrunk_*_ranking.csv`; figures `gof_pairs`, `rank_ci_vs_n` |
