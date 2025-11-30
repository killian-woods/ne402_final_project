"""
Imperial-unit Wrapper for IAPWS97 Water/Steam Properties

This module defines the `ImperialIAPWS97` class, which wraps the IAPWS97 library
to provide water and steam properties in imperial units (psia, °F, BTU/lbm, lbf,
ft/in, hr/s). It supports optional conversions for length (`use_inches`) and
time (`use_hours`) and exposes common thermophysical properties:

- T: temperature
- P: pressure
- h: specific enthalpy
- rho: density
- c_p: specific heat
- mu: dynamic viscosity
- sigma: surface tension
- x: vapor quality
- Prandt: Prandtl number
- k: thermal conductivity

Example:
    water = ImperialIAPWS97(P=1000, T=500, use_inches=True)
    print(water.rho, water.c_p, water.mu)
"""

import sys
from pathlib import Path
from functools import cached_property
from iapws import IAPWS97

# Add src folder to sys.path for local imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utilities.units.conversions import convert_units

# Valid property pairs for initializing IAPWS97
VALID_PAIRS = {
    ("P", "T"),
    ("P", "x"),
    ("T", "x"),
    ("P", "h"),
    ("T", "h"),
}


class ImperialIAPWS97:
    """
    Imperial-unit wrapper around IAPWS97 with optional inch- and hour-based conversions.

    Converts SI outputs from IAPWS97 to:
        - Pressure: psia
        - Temperature: °F
        - Enthalpy: BTU/lbm
        - Density: lbm/ft³ or lbm/in³
        - Specific heat: BTU/lbm·°F
        - Viscosity: lbm/(ft·s) or lbm/(in·hr)
        - Surface tension: lbf/ft or lbf/in
        - Thermal conductivity: BTU/(s·ft·°F) or BTU/(s·in·°F)
    """

    def __init__(
        self, P=None, T=None, h=None, x=None, use_inches=False, use_hours=False
    ):
        """
        Initialize an ImperialIAPWS97 instance with two independent thermodynamic properties.

        Args:
            P (float, optional): Pressure [psia].
            T (float, optional): Temperature [°F].
            h (float, optional): Specific enthalpy [BTU/lbm].
            x (float, optional): Vapor quality (0-1).
            use_inches (bool): Convert lengths from ft → in if True.
            use_hours (bool): Convert times from s → hr if True.

        Raises:
            ValueError: If invalid number of properties is given or pair is invalid.
        """
        self._length_unit = "in" if use_inches else "ft"
        self._time_unit = "hr" if use_hours else "s"

        self._provided = self._validate_inputs(P, T, h, x)
        self._si_args = self._convert_to_SI(P, T, h, x)
        self._water = self._instantiate_water(self._provided, self._si_args)

    # -------------------------
    # Private helper methods
    # -------------------------

    def _validate_inputs(self, P, T, h, x):
        """Ensure exactly 2 valid independent properties are provided."""
        args = {"P": P, "T": T, "h": h, "x": x}
        provided = {k: v for k, v in args.items() if v is not None}

        if len(provided) != 2:
            raise ValueError(
                f"Exactly two independent properties must be provided, got {len(provided)}: {list(provided.keys())}"
            )

        pair = tuple(sorted(provided.keys()))
        if pair not in VALID_PAIRS:
            raise ValueError(f"Invalid property pair: {pair}")

        return provided

    def _convert_to_SI(self, P, T, h, x):
        """Convert all inputs to SI units suitable for IAPWS97."""
        P_SI = convert_units(P, "pressure", "psi", "MPa") if P is not None else None
        T_SI = convert_units(T, "temperature", "F", "K") if T is not None else None
        h_SI = None
        if h is not None:
            h_SI = convert_units(h, "energy", "BTU", "kJ")
            h_SI = convert_units(h_SI, "mass", "lbm", "kg", -1, -1)
        return {"P": P_SI, "T": T_SI, "h": h_SI, "x": x}

    def _instantiate_water(self, provided, si_args):
        """
        Instantiate the IAPWS97 object, handling subcooled, two-phase, and superheated regions.
        """
        init_args = {k: si_args[k] for k in provided.keys()}
        water = IAPWS97(**init_args)

        if not isinstance(getattr(water, "cp", None), (float, int)):
            init_args["T"] = water.T
            water = IAPWS97(**init_args)

        if not isinstance(getattr(water, "cp", None), (float, int)) or water.cp is None:
            P_SI = init_args.get("P")
            h_SI = init_args.get("h")
            h_f = IAPWS97(P=P_SI, x=0).h
            h_g = IAPWS97(P=P_SI, x=1).h
            quality = (h_SI - h_f) / (h_g - h_f)

            if quality < 0:
                water = IAPWS97(P=P_SI, h=h_SI)
            elif 0 <= quality <= 1:
                water = IAPWS97(P=P_SI, x=quality)
            else:
                water = IAPWS97(P=P_SI, h=h_SI)

        return water

    # -------------------------
    # Public cached properties
    # -------------------------

    @cached_property
    def T(self) -> float:
        """Temperature [°F]."""
        return convert_units(self._water.T, "temperature", "K", "F")

    @cached_property
    def P(self) -> float:
        """Pressure [psia]."""
        return convert_units(self._water.P, "pressure", "MPa", "psia")

    @cached_property
    def h(self) -> float:
        """Specific enthalpy [BTU/lbm]."""
        metric = self._water.h
        ans = convert_units(metric, "energy", "kJ", "BTU")
        return convert_units(ans, "mass", "kg", "lbm", -1, -1)

    @cached_property
    def rho(self) -> float:
        """Density [lbm/ft³ or lbm/in³]."""
        metric = self._water.rho
        ans = convert_units(metric, "mass", "kg", "lbm")
        ans = convert_units(ans, "length", "m", self._length_unit, -3, -3)
        return ans

    @cached_property
    def x(self) -> float:
        """Vapor quality (dimensionless, 0-1)."""
        return self._water.x

    @cached_property
    def sigma(self) -> float:
        """Surface tension [lbf/ft or lbf/in]."""
        metric = self._water.sigma
        ans = convert_units(metric, "force", "N", "lbf")
        return convert_units(ans, "length", "m", self._length_unit, -1, -1)

    @cached_property
    def c_p(self) -> float:
        """Specific heat [BTU/lbm·°F]."""
        metric = self._water.cp
        ans = convert_units(metric, "energy", "kJ", "BTU")
        ans = convert_units(ans, "mass", "kg", "lbm", -1, -1)
        return convert_units(ans, "temperature", "K", "F", -1, -1, delta=True)

    @cached_property
    def mu(self) -> float:
        """Dynamic viscosity [lbm/(ft·s) or lbm/(in·hr)]."""
        metric = self._water.mu
        ans = convert_units(metric, "mass", "kg", "lbm")
        ans = convert_units(ans, "length", "m", self._length_unit, -1, -1)
        return convert_units(ans, "time", "s", self._time_unit, -1, -1)

    @cached_property
    def Prandt(self) -> float:
        """Prandtl number (dimensionless)."""
        return self._water.Prandt

    @cached_property
    def k(self) -> float:
        """Thermal conductivity [BTU/(s·ft·°F) or BTU/(s·in·°F)]."""
        metric = self._water.k
        ans = convert_units(metric, "energy", "J", "BTU")
        ans = convert_units(ans, "time", "s", self._time_unit, -1, -1)
        ans = convert_units(ans, "length", "m", self._length_unit, -1, -1)
        return convert_units(ans, "temperature", "K", "F", -1, -1, delta=True)
