from numpy import pi

from utilities.units.iapws97_imperial import ImperialIAPWS97


class Reactor:
    # given values
    core_power: float = 1.296614e10  # BTU/hr
    gamma_f: float = 0.974  # -
    pressure: int = 1040  # psia
    feed_temp: int = 410  # F
    F_q: float = 3.05
    F_z: float = 1.45
    H_core: int = 176  # in
    D_o: float = 0.4039  # in
    D_i: float = 0.3441  # in
    D_f: float = 0.3386  # in
    S: float = 0.5098  # in
    k_clad: float = 9.6 / 12  # BTU/(hr.in.F)
    H_G: float = 1200 / 144  # BTU/(hr.in^2.F)
    L_can: float = 5.52  # in
    num_assemblies: int = 872  # -
    _num_rods_per_assembly: int = 92  # -
    num_water_rods_per_assembly: int = 8  # -
    num_grids: int = 8  # -
    loss_grid: float = 0.6  # -
    loss_inlet: float = 1.5
    loss_outlet: float = 1.0
    loss_downcomer: float = 2.5
    D_v: int = 280  # in
    D_ch: int = 184  # in
    H_chimney: int = 144  # in

    # found values
    steel_abs_roughness = 1.5e-3 * 12  # in
    T_melt = 5156.33  # F

    # calculated values
    P_w: float = pi * D_o
    A: float = S**2 - (pi / 4) * (D_o**2)
    D_e: float = 4 * A / P_w  # in
    num_rods = num_assemblies * _num_rods_per_assembly
    rod_power: float = core_power * gamma_f / num_rods

    # chimney
    A_ch = pi / 4 * (D_ch**2)
    P_w_ch = pi * D_ch
    D_e_ch = D_ch

    # downcomer
    A_v = pi / 4 * (D_v**2 - D_ch**2)
    P_w_v = pi * (D_v + D_ch)
    D_e_v = 4 * A_v / P_w_v

    # objects
    inlet_water = ImperialIAPWS97(P=pressure, x=0, use_inches=True, use_hours=True)
    feed_water = ImperialIAPWS97(
        P=pressure, T=feed_temp, use_inches=True, use_hours=True
    )
