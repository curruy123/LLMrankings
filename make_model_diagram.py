"""Draws outputs/figures/btl_model_diagram.png (explanatory figure of the BTL model)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.special import expit

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
o = pd.read_csv("outputs/overall_ranking.csv", index_col=0)
s = pd.read_csv("outputs/overall_fit_summary.csv").iloc[0]
g = float(s.gamma)
A, B = "gemini-2.5-pro", "claude-opus-4-20250514"
tA, tB = o.loc[A, "theta"], o.loc[B, "theta"]
pAB = expit(tA - tB + g)

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(13, 7.6))
fig.text(0.04, 0.955, "Bradley–Terry–Luce (BTL) model for pairwise human preferences",
         fontsize=16, weight="bold", color=INK)
fig.text(0.04, 0.915, "Each LLM gets one latent strength θ; the chance a voter prefers A over B depends only on the strength gap.",
         fontsize=11, color=INK2)

# --- equation banner ---
ax0 = fig.add_axes([0.04, 0.76, 0.92, 0.13]); ax0.axis("off")
ax0.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.08",
              fc="#f3f6fb", ec=BLUE, lw=1.2, transform=ax0.transAxes))
ax0.text(0.5, 0.62, r"$P(\mathrm{A\ beats\ B}) \;=\; \sigma(\,\theta_A - \theta_B + \gamma\,)$,   "
         r"$\sigma(x)=\dfrac{1}{1+e^{-x}}$", ha="center", va="center", fontsize=17, color=INK)
ax0.text(0.5, 0.2, "θ = model strength (log-odds; constrained Σθ = 0, so θ is relative to the average model)     "
         f"γ = position bias (here γ = {g:+.3f}: position B very slightly favoured)",
         ha="center", va="center", fontsize=9.5, color=INK2)

# --- panel 1: strength scale ---
ax1 = fig.add_axes([0.04, 0.12, 0.27, 0.56])
show = [A, "o3-2025-04-16", B, "gpt-4o-mini-2024-07-18"]
lbl = {A: "Gemini 2.5 Pro (#1)", "o3-2025-04-16": "o3 (#5)", B: "Claude Opus 4 (#28)",
       "gpt-4o-mini-2024-07-18": "GPT-4o mini (#52)"}
ax1.axvline(0, color=GRID, lw=2)
th_all = o.theta.values
ax1.scatter(np.zeros_like(th_all), th_all, s=10, color="#b5b4ae", zorder=2)
for m in show:
    col = BLUE if m == A else ORANGE if m == B else INK2
    ax1.scatter(0, o.loc[m, "theta"], s=60, color=col, zorder=3)
    ax1.text(0.08, o.loc[m, "theta"], f"{lbl[m]}   θ = {o.loc[m, 'theta']:+.2f}",
             va="center", fontsize=9.5, color=col if col != INK2 else INK)
ax1.annotate("", xy=(-0.12, tA), xytext=(-0.12, tB),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
ax1.text(-0.16, (tA + tB) / 2, f"gap\n{tA - tB:.2f}", ha="right", va="center", fontsize=9.5)
ax1.set_xlim(-0.45, 1.25); ax1.set_ylim(-1.05, 1.05)
ax1.set_xticks([]); ax1.set_ylabel("estimated strength θ (52 models; grey dots)")
for sp in ["top", "right", "bottom"]: ax1.spines[sp].set_visible(False)
ax1.set_title("1. One strength per model", loc="left", weight="bold", fontsize=11)

# --- panel 2: sigmoid ---
ax2 = fig.add_axes([0.38, 0.12, 0.28, 0.56])
x = np.linspace(-2, 2, 300)
ax2.plot(x, expit(x + g), color=BLUE, lw=2.5)
ax2.axhline(0.5, color=GRID, lw=1); ax2.axvline(0, color=GRID, lw=1)
d = tA - tB
ax2.plot([d, d], [0, pAB], color=INK2, ls="--", lw=1)
ax2.plot([-2, d], [pAB, pAB], color=INK2, ls="--", lw=1)
ax2.scatter([d], [pAB], s=60, color=BLUE, zorder=3)
ax2.text(d - 0.05, pAB + 0.05, f"Gemini 2.5 Pro vs\nClaude Opus 4:\nP = σ({d:.2f} − {abs(g):.2f}) = {pAB:.2f}",
         ha="right", va="bottom", fontsize=9.5)
ax2.set_xlim(-2, 2); ax2.set_ylim(0, 1)
ax2.set_xlabel("strength gap  θ_A − θ_B"); ax2.set_ylabel("P(voter prefers A)")
for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)
ax2.grid(color=GRID, lw=0.6)
ax2.set_title("2. Gap → win probability", loc="left", weight="bold", fontsize=11)

# --- panel 3: estimation & uncertainty ---
ax3 = fig.add_axes([0.71, 0.12, 0.26, 0.56]); ax3.axis("off")
ax3.set_title("3. Estimation & uncertainty", loc="left", weight="bold", fontsize=11)
steps = [
    ("Data", f"{int(s.n_decisive):,} decisive English votes,\n{int(s.n_models)} LLMs, "
             f"{int(s.n_sessions):,} sessions"),
    ("Fit", "Maximum likelihood (Newton's method).\nCategory models = same fit on that\ncategory's votes only."),
    ("Standard errors", "Session-clustered (votes from the\nsame session may be correlated)."),
    ("Rank uncertainty", "Resample sessions 200×, refit →\n95% rank interval and P(rank 1).\n"
                         f"e.g. Gemini 2.5 Pro: #1 in {100 * o.loc[A, 'p_rank1']:.1f}% of refits"),
]
y = 0.97
for i, (h, t) in enumerate(steps):
    hgt = 0.06 + 0.055 * t.count("\n")
    ax3.add_patch(FancyBboxPatch((0.0, y - hgt - 0.065), 1, hgt + 0.06,
                  boxstyle="round,pad=0,rounding_size=0.03", fc="white", ec=GRID, lw=1.2))
    ax3.text(0.04, y - 0.02, h, weight="bold", fontsize=9.8, va="top", color=BLUE)
    ax3.text(0.04, y - 0.075, t, fontsize=9, va="top", color=INK)
    if i < len(steps) - 1:
        ax3.annotate("", xy=(0.5, y - hgt - 0.085), xytext=(0.5, y - hgt - 0.06),
                     arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    y -= hgt + 0.11

fig.text(0.04, 0.035, "Values from the fitted overall model (outputs/overall_ranking.csv). "
         "A strength gap of 0.4 ≈ a 60/40 preference.", fontsize=8.5, color=INK2)
fig.savefig("outputs/figures/btl_model_diagram.png", dpi=180, bbox_inches="tight", facecolor="white")
