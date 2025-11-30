"""
Plot Saving Utility

Provides a function to save Matplotlib figures to a specified folder
within the project structure, optionally appending a timestamp
to ensure unique filenames.

The function ensures the folder exists, resolves relative paths
from a base project directory, and saves the figure at high resolution.
"""

from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def save_plot(
    fig: plt.Figure,
    folder: Path,
    name: str = "plot",
    timestamp: bool = False,
    base: Path = Path("src"),
) -> Path:
    """
    Save a Matplotlib figure to a specified folder inside the project.

    Ensures the folder exists, optionally appends a timestamp, and
    closes the figure after saving to free memory.

    Args:
        fig (plt.Figure): The Matplotlib figure object to save.
        folder (Path): Folder path relative to `base` or absolute Path.
        name (str, optional): Base filename without extension. Defaults to "plot".
        timestamp (bool, optional): Whether to append a timestamp to the filename. Defaults to False.
        base (Path, optional): Base folder to resolve relative paths from. Defaults to Path("src").

    Returns:
        Path: Full path to the saved file.

    Raises:
        FileNotFoundError: If the `base` folder cannot be located in the parent directories.
    """
    # Resolve relative folder to project root if needed
    if not folder.is_absolute():
        current_dir = Path(__file__).parent
        while True:
            if (current_dir / base).exists():
                project_src = current_dir / base
                break
            if current_dir.parent == current_dir:
                raise FileNotFoundError(
                    f"Could not find base folder '{base}' in parent directories."
                )
            current_dir = current_dir.parent
        folder = project_src / folder

    # Ensure the folder exists
    folder.mkdir(parents=True, exist_ok=True)

    # Build filename with optional timestamp
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{ts}.png"
    else:
        filename = f"{name}.png"

    filepath = folder / filename

    # Save figure at high resolution and close to free memory
    fig.savefig(filepath, dpi=900, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to: {filepath}")
    return filepath
