"""Figure 5: From design space to Pareto frontier.

Two-panel feasible-plan cloud + Pareto frontier with serial / parallel /
finite-concurrency optima highlighted. Outputs:

  figure_planning_bw_frontier.pdf
  figure_planning_bw_frontier.png

Run from this directory:
  python3 render_planning_frontier.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

rng = np.random.default_rng(7)

# -----------------------------
# Pareto frontier
# -----------------------------
frontier = np.array([
    [1.15, 8.05],
    [2.30, 6.65],
    [3.55, 5.55],
    [5.15, 4.35],
    [7.60, 3.15],
    [10.95, 2.15],
])

parallel_pt = frontier[0]
finite_pt = frontier[3]
serial_pt = frontier[-1]

# -----------------------------
# Irregular feasible-plan cloud
# -----------------------------
x_extra = np.array([
    0.95, 1.25, 1.55, 1.80, 2.05, 2.35, 2.75,
    3.05, 3.35, 3.85, 4.15, 4.45, 4.85,
    5.35, 5.70, 6.05, 6.45, 6.85,
    7.20, 7.95, 8.35, 8.75, 9.15,
    9.65, 10.10, 10.55, 11.25,
])

frontier_y = np.interp(x_extra, frontier[:, 0], frontier[:, 1])

offsets = np.array([
    0.75, 1.55, 0.95, 2.10, 0.55, 1.35, 2.65,
    0.70, 1.85, 0.95, 2.35, 0.55, 1.65,
    0.80, 2.15, 1.15, 2.75, 0.75,
    1.55, 0.95, 2.30, 0.55, 1.75,
    0.85, 1.95, 0.60, 1.25,
])

x_jitter = rng.normal(0, 0.08, size=x_extra.size)
y_jitter = rng.normal(0, 0.13, size=x_extra.size)

extra_points = np.column_stack([
    x_extra + x_jitter,
    frontier_y + offsets + y_jitter,
])

extra_points[:, 1] = np.clip(extra_points[:, 1], 1.65, 8.85)

# Manual label-clearance filtering. These boxes correspond to the right
# panel's annotation text areas; points inside them are dropped so labels
# do not overlap grey dots.
clear_boxes = [
    (1.85, 4.15, 8.25, 8.95),    # parallel label area
    (6.15, 9.40, 4.75, 5.55),    # finite-concurrency label area
    (8.95, 11.30, 3.05, 3.65),   # serial label area
]


def inside_any_clear_box(p):
    x, y = p
    for x0, x1, y0, y1 in clear_boxes:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


extra_points = np.array([p for p in extra_points if not inside_any_clear_box(p)])

# Replacement points in visually safe regions to keep cloud density.
safe_extra = np.array([
    [2.85, 7.55],
    [4.95, 6.25],
    [6.75, 6.15],
    [7.95, 4.25],
    [9.65, 4.45],
    [10.35, 2.75],
])

points = np.vstack([extra_points, safe_extra, frontier])
rng.shuffle(points)

B = points[:, 0]
W = points[:, 1]

# -----------------------------
# Style
# -----------------------------
colors = {
    "axis": "#1F2937",
    "points": "#A7AFBA",
    "frontier": "#263241",
    "parallel": "#D55E00",
    "serial": "#0072B2",
    "finite": "#009E73",
    "arrow": "#263241",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=False)

xmin, xmax = 0.55, 11.75
ymin, ymax = 1.35, 9.15


def style_axis(ax):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("B — slowest-request latency")
    ax.set_ylabel("W — aggregate time")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(colors["axis"])
    ax.spines["bottom"].set_color(colors["axis"])
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.tick_params(length=0)


def draw_axis_arrows(ax):
    ax.annotate("", xy=(xmax, ymin), xytext=(xmin, ymin),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=colors["axis"],
                                shrinkA=0, shrinkB=0, mutation_scale=12),
                clip_on=False)
    ax.annotate("", xy=(xmin, ymax), xytext=(xmin, ymin),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=colors["axis"],
                                shrinkA=0, shrinkB=0, mutation_scale=12),
                clip_on=False)


def draw_points(ax, alpha=0.9):
    ax.scatter(B, W, s=58, color=colors["points"], edgecolor="white",
               linewidth=0.8, alpha=alpha, zorder=2)


label_box = dict(boxstyle="round,pad=0.18,rounding_size=0.08",
                 facecolor="white", edgecolor="none", alpha=0.86)

# -----------------------------
# Left panel
# -----------------------------
ax = axes[0]
style_axis(ax)
draw_axis_arrows(ax)
ax.set_title("(a) Feasible plans", loc="left", pad=10, fontweight="bold")
draw_points(ax, alpha=0.90)

# -----------------------------
# Right panel
# -----------------------------
ax = axes[1]
style_axis(ax)
draw_axis_arrows(ax)
ax.set_title("(b) Pareto frontier", loc="left", pad=10, fontweight="bold")

draw_points(ax, alpha=0.62)

ax.plot(frontier[:, 0], frontier[:, 1], color=colors["frontier"],
        linewidth=3.1, solid_capstyle="round", solid_joinstyle="round", zorder=4)

ax.scatter(frontier[1:-1, 0], frontier[1:-1, 1], s=82,
           color=colors["frontier"], edgecolor="white", linewidth=1.0, zorder=5)

ax.scatter(parallel_pt[0], parallel_pt[1], marker="s", s=165,
           color=colors["parallel"], edgecolor="white", linewidth=1.0, zorder=7)
ax.scatter(serial_pt[0], serial_pt[1], marker="s", s=165,
           color=colors["serial"], edgecolor="white", linewidth=1.0, zorder=7)
ax.scatter(finite_pt[0], finite_pt[1], marker="*", s=380,
           color=colors["finite"], edgecolor="white", linewidth=1.1, zorder=8)

# Iso-cost lines
slope = -0.55
xline = np.linspace(xmin, xmax, 240)
yline = finite_pt[1] + slope * (xline - finite_pt[0])
ax.plot(xline, yline, linestyle=(0, (4, 4)), color=colors["finite"],
        linewidth=1.45, alpha=0.75, zorder=1)
for offset in [-0.85, 0.85]:
    ax.plot(xline, yline + offset, linestyle=(0, (4, 4)),
            color=colors["finite"], linewidth=1.05, alpha=0.28, zorder=1)

ax.annotate(r"$\pi^*_{\mathrm{parallel}}$", xy=parallel_pt, xytext=(2.05, 8.55),
            color=colors["parallel"], fontsize=17, fontweight="bold", bbox=label_box,
            arrowprops=dict(arrowstyle="-", color=colors["parallel"], lw=1.1,
                            shrinkA=2, shrinkB=4),
            zorder=10)
ax.annotate(r"$\arg\min_\pi T_P(\pi)$", xy=finite_pt, xytext=(6.65, 5.22),
            color=colors["finite"], fontsize=17, fontweight="bold", bbox=label_box,
            arrowprops=dict(arrowstyle="-", color=colors["finite"], lw=1.1,
                            shrinkA=2, shrinkB=5),
            zorder=10)
ax.annotate(r"$\pi^*_{\mathrm{serial}}$", xy=serial_pt, xytext=(9.10, 3.35),
            color=colors["serial"], fontsize=17, fontweight="bold", bbox=label_box,
            arrowprops=dict(arrowstyle="-", color=colors["serial"], lw=1.1,
                            shrinkA=2, shrinkB=4),
            zorder=10)

# WaRP arrow between panels — wider wspace + larger end-gap so the
# arrow head sits clearly inside the gutter, not jammed against the
# right panel's y-axis.
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88, wspace=0.52)
fig.canvas.draw()

left_pos = axes[0].get_position()
right_pos = axes[1].get_position()
x_start = left_pos.x1 + 0.025
x_end = right_pos.x0 - 0.045
y_mid = (left_pos.y0 + left_pos.y1) / 2

arrow = FancyArrowPatch((x_start, y_mid), (x_end, y_mid),
                        transform=fig.transFigure, arrowstyle="-|>",
                        mutation_scale=18, linewidth=2.0, color=colors["arrow"])
fig.add_artist(arrow)

fig.text((x_start + x_end) / 2, y_mid + 0.035, "WaRP",
         ha="center", va="bottom", fontsize=17, fontweight="bold",
         color=colors["arrow"])

import os
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, "figure_planning_bw_frontier.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(out_dir, "figure_planning_bw_frontier.png"), dpi=350, bbox_inches="tight")
print(f"saved: {out_dir}/figure_planning_bw_frontier.{{pdf,png}}")
