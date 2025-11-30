from pathlib import Path

NUM_PLOT_POINTS = (
    500  # num points for plots. increase a lot when creating final plots for project
)
FIGS_PATH = Path(__file__).parent.parent / "figs"
CRIT_FIGS_PATH = FIGS_PATH / "crit"
HOT_FIGS_PATH = FIGS_PATH / "hot"
AVG_FIGS_PATH = FIGS_PATH / "avg"
BOTH_FIGS_PATH = FIGS_PATH / "both"
