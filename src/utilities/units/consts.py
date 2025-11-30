"""
Physical Constants for Reactor Analysis

This module provides fundamental physical constants used throughout
the reactor thermal-hydraulics calculations, expressed in Imperial units.
"""

# Gravitational conversion factor (mass-to-force)
G_C = 32.174  # (lbm*ft)/(lbf*s^2)
G_C_in_hr = G_C * 12 * (3600**2)  # (lbm*in)/(lbf*hr^2)

# Standard acceleration due to gravity
G = 32.174  # ft/s^2
G_in_hr = G * 12 * (3600**2)  # in/hr^2
