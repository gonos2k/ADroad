"""Boundary layer: conductance (BLC), aerodynamic resistance, latent heat.
Karsisto 2024, eq 11-26. Scalar (single point) at M1.

BLC is an iterative fixed point (stability correction). Two variants (P0 §5):
  - early_stop=True  → BLC-v1: exact early-stop, matches RoadSurf-Python (parity)
  - early_stop=False → BLC-v0: fixed 40 unroll (for gradient smoke tests)

Domain-sensitive ops (sqrt/log/exp) go through registered guarded wrappers.
"""

from __future__ import annotations

from .branches import guarded_sqrt, guarded_log, guarded_exp

MAX_ITER = 40
CONV_LIM = 0.001


def calc_raero(logMom, logHeat, PSIM, PSIH, VK, VZ):
    r = (logMom + PSIM) * (logHeat + PSIH) / (VK * VK * VZ)
    return 30.0 if r > 30.0 else r          # reference caps at 30 s/m


def _sat_vapor(T):
    # over ice below 0, over water otherwise (eq 22-23)
    if T < 0:
        return 0.61078 * guarded_exp("boundary_layer.sat_vapor", 21.875 * T / (T + 265.5))
    return 0.61078 * guarded_exp("boundary_layer.sat_vapor", 17.269 * T / (T + 237.3))


def calc_le(Tsurf, Tair, Rhz, AirDens, AirHCap, Raero, LVap, LFus, WatDen, dt, SrfWat):
    TaK = Tair + 273.15
    PsychC = 0.1 * (0.00063 * TaK + 0.47496)
    ESurf = _sat_vapor(Tsurf)
    EAir = min(0.01 * Rhz, 1.0) * _sat_vapor(Tair)
    LE = (AirDens * AirHCap * (ESurf - EAir)) / (PsychC * Raero)
    if Tsurf >= 0.0:
        Evap = (LE / (LVap * WatDen)) * 1000.0 * dt
    else:
        Evap = (LE / (LFus * WatDen)) * 1000.0 * dt
    if LE > 0.0 and SrfWat <= 0.0:          # no water to evaporate
        LE, Evap = 0.0, 0.0
    return LE, Evap


def calc_blc_and_le(Tsurf, Tair, VZ, Rhz, BLCond_init, SrfWat, dt, phy, early_stop=True):
    """Returns (BLCond, LE_Flux, EvapmmTS)."""
    TaK = Tair + 273.15
    AirDens = 100000.0 / (287.05 * TaK)
    AirHCap = 1005.0 + ((TaK - 250.0) ** 2) / 3364.0
    AirVCap = AirHCap * AirDens
    WatDen = -0.0050 * Tsurf * Tsurf + 0.0079 * Tsurf + 1000.0028

    PSIM = 0.0
    PSIH = 0.0
    BLCond = BLCond_init
    for j in range(1, MAX_ITER + 1):
        BLCond_old = BLCond
        UStar = phy["VK"] * VZ / (phy["logUstar"] + PSIM)
        BLCond = AirVCap * phy["VK"] * UStar / (phy["logCond"] + PSIH)
        Stab = (-phy["VK"] * phy["ZRefT"] * phy["Grav"] * BLCond * (Tsurf - Tair)
                / (AirVCap * (Tair + 273.15) * (UStar ** 3)))
        if Stab > 1:
            Stab = 1
        if Stab > 0:                        # stable
            PSIH = 4.7 * Stab
            PSIM = PSIH
        else:                               # unstable
            arg = (1.0 + guarded_sqrt("boundary_layer.psi_unstable", 1.0 - 16.0 * Stab)) / 2.0
            PSIH = -2.0 * guarded_log("boundary_layer.psi_unstable", arg)
            PSIM = 0.6 * PSIH
        if early_stop and abs(BLCond - BLCond_old) < CONV_LIM and j >= 5:
            break

    Raero = calc_raero(phy["logMom"], phy["logHeat"], PSIM, PSIH, phy["VK"], VZ)
    LE, Evap = calc_le(Tsurf, Tair, Rhz, AirDens, AirHCap, Raero,
                       phy["LVap"], phy["LFus"], WatDen, dt, SrfWat)
    return BLCond, LE, Evap
