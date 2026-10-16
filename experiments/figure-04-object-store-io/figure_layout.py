"""Shared physical layout for the two Figure 3 assets."""

FIGURE_HEIGHT_IN = 1.52
LEFT_WIDTH_IN = 1.04
RIGHT_WIDTH_IN = LEFT_WIDTH_IN * (0.66 / 0.32)

# Both assets use the same normalized row geometry. Their physical width
# ratio matches the LaTeX subfigure ratio, so the rows align after inclusion.
AXES_TOP = 0.83
AXES_BOTTOM = 0.21
AXES_HSPACE = 0.60
X_LABEL_Y = 0.035
