import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

BLUE = "#1F4E79"; LBLUE = "#D5E8F0"; GREY = "#F2F2F2"
GREEN = "#2E7D32"; RED = "#C62828"; ORANGE = "#E8A33D"

# ---- Fig 1: architecture pipeline ----
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=130)
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
layers = [
    ("ledger  (mass-audit, immutable record)", "residual = code-leak detector"),
    ("deviation budget  (residual gate + diagnostics)", "P0: max|r| <= 1e-9 ; burden counts"),
    ("skill gate  (RMSE hard gate + burden guardrail)", "MAE / freeze-thaw = report-only"),
    ("forecast DA report  (state / parameter DA)", "report-only + machine-readable sidecar"),
]
y = 6.6
for i, (title, sub) in enumerate(layers):
    box = FancyBboxPatch((0.6, y), 8.8, 1.05, boxstyle="round,pad=0.04,rounding_size=0.12",
                         linewidth=1.4, edgecolor=BLUE, facecolor=LBLUE if i % 2 == 0 else GREY)
    ax.add_patch(box)
    ax.text(5.0, y + 0.68, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(5.0, y + 0.28, sub, ha="center", va="center", fontsize=8.2, color="#444444", style="italic")
    if i < 3:
        ax.add_patch(FancyArrowPatch((5.0, y), (5.0, y - 0.55), arrowstyle="-|>",
                     mutation_scale=16, linewidth=1.6, color=BLUE))
    y -= 1.62
ax.text(5.0, 0.15, "each boundary performs defensive validation (LedgerError / SkillError)",
        ha="center", fontsize=8.2, color="#666666")
plt.tight_layout(pad=0.3)
plt.savefig("fig_arch.png", bbox_inches="tight"); plt.close()

# ---- Fig 2: forecast DA cycle schematic ----
fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=130)
ax.set_xlim(0, 12); ax.set_ylim(-2.4, 3.2); ax.axis("off")
# phase bands
ax.axvspan(0, 3, color=GREY); ax.axvspan(3, 5, color=LBLUE); ax.axvspan(5, 12, color="#FBF3E4")
for x, t in [(1.5, "spin to k0\n(background x_b)"), (4.0, "assimilation window\nfit dx  (min J)"),
             (8.5, "forecast lead\nFREE-RUN (no obs)")]:
    ax.text(x, 2.75, t, ha="center", va="center", fontsize=9, fontweight="bold", color=BLUE)
ax.axhline(0, color="#999999", lw=0.8)
# analysis vs background trajectories over lead
xl = np.linspace(5, 11.6, 60)
bg = 0.9 * np.sin((xl - 5) * 0.7) + 0.55
da = 0.9 * np.sin((xl - 5) * 0.7) + 0.08
ax.plot(xl, da, color=GREEN, lw=2.2, label="DA (state)  -> lower RMSE")
ax.plot(xl, bg, color=RED, lw=2.0, ls="--", label="no-DA (background)")
# obs points
xo = np.linspace(5.2, 11.4, 12)
yo = 0.9 * np.sin((xo - 5) * 0.7)
ax.scatter(xo, yo, s=22, color="#333333", zorder=5, label="observations")
# dx correction arrow at window end
ax.annotate("", xy=(5.05, 0.08), xytext=(5.05, 0.55),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8))
ax.text(5.15, 0.33, "dx", fontsize=9, color=BLUE, fontweight="bold")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=3, fontsize=8.2, frameon=False)
ax.text(4.0, -1.0, "state control:  layers 1:5 offset", ha="center", fontsize=8, color="#555555")
plt.tight_layout(pad=0.3)
plt.savefig("fig_cycle.png", bbox_inches="tight"); plt.close()

# ---- Fig 3: single-window forecast RMSE ----
fig, ax = plt.subplots(figsize=(5.4, 2.9), dpi=130)
names = ["constant_initial", "no_DA\n(background)", "DA (state)"]
vals = [0.9180, 0.2210, 0.2082]
cols = ["#9E9E9E", RED, GREEN]
b = ax.bar(names, vals, color=cols, width=0.6, edgecolor="white")
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("forecast RMSE (°C)"); ax.set_ylim(0, 1.03)
ax.set_title("Single-window forecast DA (k0=2000, lead 480)", fontsize=10, color=BLUE)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.3)
plt.savefig("fig_single.png", bbox_inches="tight"); plt.close()

# ---- Fig 4: multi-window delta RMSE ----
fig, ax = plt.subplots(figsize=(5.4, 2.9), dpi=130)
k0 = ["1500", "2100", "2700", "3300"]
d = [0.1183, 0.0200, -0.0046, -0.0385]
cols = [RED if x > 0 else GREEN for x in d]
b = ax.bar(k0, d, color=cols, width=0.6, edgecolor="white")
for r, v in zip(b, d):
    ax.text(r.get_x() + r.get_width() / 2, v + (0.006 if v >= 0 else -0.010),
            f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, fontweight="bold")
ax.axhline(0, color="#333333", lw=1.0)
ax.set_ylabel("Δ RMSE  (DA − no-DA)"); ax.set_xlabel("analysis window start  k0")
ax.set_title("Multi-window reproduction: DA wins 2/4  (REPORT_ONLY)", fontsize=9.5, color=BLUE)
ax.text(0.5, 0.128, "worse (DA loses)", color=RED, fontsize=8)
ax.text(2.5, -0.052, "better (DA wins)", color=GREEN, fontsize=8, ha="center")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.3)
plt.savefig("fig_multi.png", bbox_inches="tight"); plt.close()

print("wrote fig_arch.png fig_cycle.png fig_single.png fig_multi.png")
