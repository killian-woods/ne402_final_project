"""
Utilities for nicely formatted answers with significant figures.

This module provides a single function, `print_answer`, which prints
a labeled answer with a specified number of significant figures and optional units.
It supports scientific notation formatting if desired.
"""

import math


def print_answer(
    question: str,
    value: float | str,
    units: str = "",
    sigfigs: int = 6,
    use_scientific: bool = False,
) -> None:
    """
    Print a formatted answer with significant figures for a given question.

    Args:
        question (str): Identifier for the question (e.g., "2", "3.a", "4.b.ii").
        value (float | str): The computed answer (numeric) or a textual response.
        units (str, optional): Units to display after the value (e.g., "m/s", "MW"). Defaults to "".
        sigfigs (int, optional): Number of significant figures for numeric values. Defaults to 6.
        use_scientific (bool, optional): If True, format numeric values in scientific notation. Defaults to False.

    Notes:
        - If `value` is a string, it is printed as-is with the units.
        - Numeric values respect the requested number of significant figures.
        - A dash "[-]" is printed when no units are provided.
    """

    def format_sigfigs(num: float, sigfigs: int, scientific: bool) -> str:
        """Format a numeric value to the requested number of significant figures."""
        if num == 0:
            return "0"
        if scientific:
            return f"{num:.{sigfigs - 1}e}"
        magnitude = int(math.floor(math.log10(abs(num))))
        decimals = sigfigs - magnitude - 1
        if decimals >= 0:
            return f"{num:.{decimals}f}"
        else:
            return f"{num:.{sigfigs}g}"

    if isinstance(value, (float, int)):
        formatted_value = format_sigfigs(value, sigfigs, use_scientific)
    else:
        formatted_value = str(value)

    unit_str = f" [{units}]" if units else "[-]"
    print(f"Question {question}: {formatted_value}{unit_str}")
