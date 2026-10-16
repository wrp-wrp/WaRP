from __future__ import annotations

from typing import Any


PAPER_SANS_FONTS = [
    "Arial",
    "Arimo",
    "Helvetica",
    "Liberation Sans",
    "Nimbus Sans",
    "Nimbus Sans L",
    "DejaVu Sans",
]

GLOBAL_PAPER_FONT_SCALE = 0.92
PAPER_LEGEND_EDGE = "#b8b8b8"
PAPER_LEGEND_LINEWIDTH = 0.45

PAPER_STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "dense_grid": {
        "font_size": 7.2,
        "axes_label_size": 7.7,
        "axes_title_size": 7.6,
        "legend_size": 6.8,
        "xtick_size": 6.8,
        "ytick_size": 6.8,
        "axes_linewidth": 0.85,
        "grid_color": "#d2d6db",
        "grid_alpha": 0.40,
        "grid_linewidth": 0.50,
        "size_scale": 1.0,
    },
    "single_col": {
        "font_size": 10.0,
        "axes_label_size": 10.8,
        "axes_title_size": 10.4,
        "legend_size": 8.8,
        "xtick_size": 9.2,
        "ytick_size": 9.2,
        "axes_linewidth": 0.90,
        "grid_color": "#d2d6db",
        "grid_alpha": 0.44,
        "grid_linewidth": 0.55,
        "size_scale": 1.0,
    },
    "single_col_microbench": {
        "font_size": 6.4,
        "axes_label_size": 6.6,
        "axes_title_size": 7.0,
        "legend_size": 5.9,
        "xtick_size": 5.8,
        "ytick_size": 5.8,
        "axes_linewidth": 0.65,
        "grid_color": "#d2d6db",
        "grid_alpha": 0.44,
        "grid_linewidth": 0.55,
        "size_scale": 1.0,
    },
    "double_col": {
        "font_size": 9.6,
        "axes_label_size": 10.2,
        "axes_title_size": 10.0,
        "legend_size": 8.6,
        "xtick_size": 8.8,
        "ytick_size": 8.8,
        "axes_linewidth": 0.90,
        "grid_color": "#d2d6db",
        "grid_alpha": 0.44,
        "grid_linewidth": 0.55,
        "size_scale": 1.0,
    },
    "tall_narrow": {
        "font_size": 9.4,
        "axes_label_size": 10.2,
        "axes_title_size": 10.0,
        "legend_size": 8.4,
        "xtick_size": 8.8,
        "ytick_size": 8.8,
        "axes_linewidth": 0.90,
        "grid_color": "#d2d6db",
        "grid_alpha": 0.44,
        "grid_linewidth": 0.55,
        "size_scale": 1.0,
    },
}


def apply_paper_style(
    plt,
    *,
    font_size: float,
    axes_label_size: float | None = None,
    axes_title_size: float | None = None,
    legend_size: float | None = None,
    xtick_size: float | None = None,
    ytick_size: float | None = None,
    axes_linewidth: float = 0.85,
    grid_color: str | None = None,
    grid_alpha: float | None = None,
    grid_linewidth: float | None = None,
    size_scale: float = 1.12,
    extra_rc: dict[str, Any] | None = None,
) -> None:
    effective_scale = size_scale * GLOBAL_PAPER_FONT_SCALE
    scaled_font_size = font_size * effective_scale
    scaled_axes_label_size = (axes_label_size if axes_label_size is not None else font_size) * effective_scale
    scaled_axes_title_size = (axes_title_size if axes_title_size is not None else font_size) * effective_scale
    scaled_legend_size = (legend_size if legend_size is not None else font_size) * effective_scale
    scaled_xtick_size = (xtick_size if xtick_size is not None else font_size) * effective_scale
    scaled_ytick_size = (ytick_size if ytick_size is not None else font_size) * effective_scale
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": PAPER_SANS_FONTS,
        "mathtext.fontset": "stixsans",
        "mathtext.default": "regular",
        "font.size": scaled_font_size,
        "axes.labelsize": scaled_axes_label_size,
        "axes.titlesize": scaled_axes_title_size,
        "legend.fontsize": scaled_legend_size,
        "xtick.labelsize": scaled_xtick_size,
        "ytick.labelsize": scaled_ytick_size,
        "axes.linewidth": axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "axes.titleweight": "regular",
        "axes.labelweight": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if grid_color is not None:
        rc["grid.color"] = grid_color
    if grid_alpha is not None:
        rc["grid.alpha"] = grid_alpha
    if grid_linewidth is not None:
        rc["grid.linewidth"] = grid_linewidth
    if extra_rc:
        rc.update(extra_rc)
    plt.rcParams.update(rc)


def apply_paper_preset(
    plt,
    preset: str,
    *,
    extra_rc: dict[str, Any] | None = None,
    **overrides: Any,
) -> None:
    if preset not in PAPER_STYLE_PRESETS:
        raise ValueError(f"unknown paper style preset: {preset}")
    params = dict(PAPER_STYLE_PRESETS[preset])
    params.update(overrides)
    if extra_rc is not None:
        merged_extra = dict(params.get("extra_rc", {}))
        merged_extra.update(extra_rc)
        params["extra_rc"] = merged_extra
    apply_paper_style(plt, **params)


def style_legend_frame(legend) -> None:
    """Apply the shared paper legend frame without changing its content."""
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor(PAPER_LEGEND_EDGE)
    frame.set_linewidth(PAPER_LEGEND_LINEWIDTH)
    frame.set_alpha(1.0)
