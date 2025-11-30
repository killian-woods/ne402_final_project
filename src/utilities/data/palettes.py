"""
Color Palette Utilities

Provides reusable color palettes and cycling functionality for plotting
and visualization. Supports consistent styling across multiple figures by
allowing colors to be cycled automatically.

Includes base palette class and several predefined palette variants.

Palettes:
    - DefaultPalette: Core high-contrast colors.
    - PastelPalette: Softer, visually pleasing pastel tones.
    - BoldPalette: Strong, high-impact colors suitable for emphasis.
    - MutedPalette: Subdued colors for background or secondary elements.
"""

from itertools import cycle

# -----------------------------
# Predefined Color Sets
# -----------------------------

# Core high-contrast colors
BASE_COLORS = [
    "#D6282A",  # Imperial Vermilion
    "#1E77B4",  # Azure Cerulean
    "#E09F3E",  # Golden Amber
    "#2E8B57",  # Celadon Green
    "#6A1B9A",  # Royal Purple
    "#1D1D1D",  # Ink Black
]

# Softer pastel colors
PASTEL_COLORS = [
    "#F0665C",  # Coral Rose
    "#4DA3E0",  # Clear Sky Blue
    "#F5B642",  # Golden Marigold
    "#66A5A1",  # Dusty Teal
    "#65C18C",  # Fresh Mint Green
    "#F48FB1",  # Rose Quartz
    "#B57EDC",  # Soft Amethyst
    "#242424",  # Graphite Black
]

# Strong, bold colors for emphasis
BOLD_COLORS = [
    "#D62828",  # Crimson Red
    "#264653",  # Dark Cyan-Blue
    "#FF9F1C",  # Bright Orange
    "#2A9D8F",  # Teal
    "#E9C46A",  # Golden Yellow
    "#6A4C93",  # Deep Purple
    "#000000",  # Black
]

# Subdued, muted colors for secondary or background elements
MUTED_COLORS = [
    "#F7A6A0",  # Soft Coral
    "#1F4E79",  # Deep Steel Blue
    "#B2DF8A",  # Soft Green
    "#A6CEE3",  # Light Cyan
    "#38755B",  # Teal-Green
    "#4A4A4A",  # Dark Gray
]

# -----------------------------
# Palette Classes
# -----------------------------


class Palette:
    """
    Base class for reusable color palettes.

    Provides functionality to cycle through a set of colors and
    track the current selection. Subclasses should define the COLORS attribute.

    Attributes:
        COLORS (list[str]): List of hex color codes used in the palette.
    """

    COLORS: list[str] = []  # Subclasses should override

    def __init__(self):
        """Initialize the color cycler using the predefined COLORS list."""
        if not self.COLORS:
            raise ValueError("Palette subclass must define a non-empty COLORS list.")
        self._cycler = cycle(self.COLORS)
        self._current: str | None = None  # Tracks the last returned color

    def next(self) -> str:
        """
        Return the next color in the palette and update the current color.

        Returns:
            str: A hex color code representing the next color.
        """
        self._current = next(self._cycler)
        return self._current

    def current(self) -> str | None:
        """
        Return the currently selected color without advancing the cycle.

        Returns:
            str | None: The current hex color code, or None if `next()` hasn't been called yet.
        """
        return self._current


# -----------------------------
# Predefined Palette Variants
# -----------------------------


class DefaultPalette(Palette):
    """Default reusable color palette for consistent plotting."""

    COLORS = BASE_COLORS


class PastelPalette(Palette):
    """Pastel palette variant with slightly higher saturation for line visibility."""

    COLORS = PASTEL_COLORS


class BoldPalette(Palette):
    """Bold palette variant with strong colors for emphasis and contrast."""

    COLORS = BOLD_COLORS
