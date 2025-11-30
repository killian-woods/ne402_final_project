"""
Plot Formatting Utility

Provides a single helper function `format_plot` to apply consistent, academically
professional styling to Matplotlib plots. This includes font settings, grid lines,
tick formatting, legend handling, logarithmic scaling, and line widths.

Intended for use across all plots in the reactor analysis or similar scientific projects.
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams


def format_plot(
    ax: plt.Axes,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    grid: bool = True,
    grid_alpha: float = 0.3,
    legend: bool = True,
    fontsize: int = 12,
    title_fontsize: int = 14,
    tight_layout: bool = True,
    loglog: bool = False,
    line_width: float = 2.5,
) -> None:
    """
    Apply consistent styling to a Matplotlib Axes.

    This function configures fonts, gridlines, axis labels, tick marks, legend display,
    line widths, and optionally log-log scaling. It enforces an academic, professional
    style for all plots.

    Args:
        ax (plt.Axes): The axes object to format.
        title (str, optional): Title of the plot. Defaults to None.
        xlabel (str, optional): Label for the X-axis. Defaults to None.
        ylabel (str, optional): Label for the Y-axis. Defaults to None.
        grid (bool, optional): Whether to display a grid. Defaults to True.
        grid_alpha (float, optional): Transparency of grid lines. Defaults to 0.3.
        legend (bool, optional): Whether to display a legend. Defaults to True.
        fontsize (int, optional): Font size for axis labels and ticks. Defaults to 12.
        title_fontsize (int, optional): Font size for plot title. Defaults to 14.
        tight_layout (bool, optional): Whether to apply tight layout to figure. Defaults to True.
        loglog (bool, optional): If True, set both axes to logarithmic scale. Defaults to False.
        line_width (float, optional): Line width for all lines in the axes. Defaults to 2.5.

    Side Effects:
        Modifies the given Axes object in-place. Can also adjust the figure layout.
    """
    # Set global font to Times New Roman
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman"]

    # Set plot title and axis labels
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontname="Times New Roman")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontname="Times New Roman")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontname="Times New Roman")

    # Configure major and minor ticks
    ax.tick_params(axis="both", which="major", labelsize=fontsize, direction="out")
    ax.tick_params(axis="both", which="minor", direction="out")
    ax.minorticks_on()

    # Configure grid lines
    if grid:
        ax.grid(True, which="major", alpha=grid_alpha, linestyle="--")
        ax.grid(True, which="minor", alpha=grid_alpha / 2, linestyle=":")

    # Show legend if requested
    if legend:
        ax.legend(fontsize=fontsize)

    # Apply log-log scaling if requested
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Adjust line widths for all lines in the axes
    for line in ax.get_lines():
        line.set_linewidth(line_width)

    # Apply tight layout if requested
    if tight_layout:
        ax.figure.tight_layout()
