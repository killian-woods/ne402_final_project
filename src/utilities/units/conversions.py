"""
Unit Conversion Utilities

Provides constants and a conversion function for converting values between
different units of the same type (length, mass, volume, temperature, force,
pressure, energy, and time).

Supports:
- Absolute and delta (differences) temperature conversions
- Unit powers for area, volume, or other exponentiated quantities
- Numpy arrays and scalar values

Example:
    convert_units(1, unit_type="length", input_unit="m", output_unit="ft")
"""

from typing import Union
import numpy as np

# Conversion factors relative to SI base units
UNIT_FACTORS = {
    "length": {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
        "km": 1000,
        "ft": 0.3048,
        "in": 0.0254,
    },
    "mass": {
        "kg": 1.0,
        "g": 0.001,
        "lb": 0.453592,
        "lbm": 0.453592,
        "oz": 0.0283495,
    },
    "volume": {
        "m3": 1.0,
        "cm3": 1e-6,
        "L": 0.001,
        "mL": 1e-6,
        "gal": 0.00378541,
    },
    "temperature": {
        "C": ("C", 1.0, 0.0),
        "K": ("C", 1.0, -273.15),
        "F": ("C", 5 / 9, -32 * 5 / 9),
        "R": ("C", 5 / 9, -491.67 * 5 / 9),
    },
    "force": {
        "N": 1.0,
        "kN": 1000.0,
        "dyn": 1e-5,
        "lbf": 4.44822,
        "kgf": 9.80665,
    },
    "time": {
        "s": 1.0,
        "min": 60.0,
        "hr": 3600.0,
        "day": 86400.0,
        "year": 31536000.0,
    },
    "pressure": {
        "Pa": 1.0,
        "kPa": 1e3,
        "MPa": 1e6,
        "bar": 1e5,
        "atm": 101325.0,
        "psi": 6894.76,
        "psia": 6894.76,
        "psf": 47.8802777778,
        "torr": 133.322,
    },
    "energy": {
        "J": 1.0,
        "kJ": 1e3,
        "MJ": 1e6,
        "cal": 4.184,
        "kcal": 4184.0,
        "Wh": 3600.0,
        "kWh": 3.6e6,
        "eV": 1.60218e-19,
        "BTU": 1055.06,
    },
}


def convert_units(
    value: Union[float, np.ndarray],
    unit_type: str = "length",
    input_unit: str = "m",
    output_unit: str = "m",
    input_power: int = 1,
    output_power: int = 1,
    delta: bool = False,  # True if converting a temperature difference (deltaT)
) -> Union[float, np.ndarray]:
    """
    Convert a value or array between units of the same type.

    Handles absolute and delta temperature conversions, as well as multiplicative
    units with powers (e.g., converting m^2 to ft^2).

    Args:
        value: Numeric value or array to convert.
        unit_type: Type of unit (length, mass, volume, temperature, force, etc.)
        input_unit: Current unit of the value.
        output_unit: Desired output unit.
        input_power: Power of the input unit (default 1).
        output_power: Power of the output unit (default 1).
        delta: If True, treat value as a temperature difference (deltaT),
            ignoring offsets but respecting multipliers.

    Returns:
        Converted value as scalar or np.ndarray, matching the input type.
    """
    factors = UNIT_FACTORS.get(unit_type)
    if factors is None:
        raise ValueError(f"Unknown unit type: {unit_type}")
    if input_unit not in factors or output_unit not in factors:
        raise ValueError(f"Unknown unit: {input_unit} or {output_unit}")

    val_arr = np.array(value, dtype=float)

    if unit_type == "temperature":
        input_base, input_mul, input_off = factors[input_unit]
        output_base, output_mul, output_off = factors[output_unit]

        if delta:
            # Only scale for temperature differences
            scale = (input_mul**input_power) / (output_mul**output_power)
            output_value = val_arr * scale
        else:
            # Absolute temperatures: apply multiplier and offset
            val_in_base = val_arr * input_mul + input_off
            output_value = (val_in_base - output_off) / output_mul
    else:
        # Other units: simple multiplicative conversion with powers
        input_factor = factors[input_unit]
        output_factor = factors[output_unit]
        base_value = val_arr * (input_factor**input_power)
        output_value = base_value / (output_factor**output_power)

    return output_value.item() if np.isscalar(value) else output_value
