"""Ground heat conduction — dry thermal core (Karsisto 2024, eq 27-31).

Explicit forward-difference scheme (no implicit solver). Pure functions on
NumPy arrays (M1 backend-neutral). Written to match RoadSurf-Python
`BalanceModel.CalcHCapHCond / calcCapDZCondDZ / calcProfile` element-wise, so
python_compat G0 parity holds.

Layer indexing (matches reference):
  Tmp / TmpNw : length NLayers+2  (0=air, 1..NLayers=layers, NLayers+1=bottom)
  ZDpth       : length NLayers+1
  CC, WCont   : length NLayers
"""

from __future__ import annotations

import numpy as np

from .branches import safe_where


def calc_hcap_hcond(TmpNw, WCont, CC, ZDpth, NLayers, DTSecs, phy, BLCond):
    """Volumetric heat capacity (VSH), heat capacity in intensity units (HS),
    and layer conductance (GCond). Returns (VSH, HS, GCond)."""
    t = TmpNw[1:NLayers + 1]                       # layer temperatures
    t2 = t * t

    # water (T>=0) vs ice (T<0) density & specific heat
    roo_water = -0.0050 * t2 + 0.0079 * t + 1000.0028
    cw_water = (0.0000102 * t2 * t2 - 0.0017169 * t2 * t
                + 0.11516 * t2 - 3.4739 * t + 4217.2)
    roo = safe_where("heat_capacity.water_ice_props", t >= 0, roo_water, 920.0)
    cw = safe_where("heat_capacity.water_ice_props", t >= 0, cw_water, 2100.0)
    chwt = roo * cw                                # volumetric heat cap of water

    # dry-ground volumetric heat capacity: layers 0,1 use asphalt; rest ground
    vsh_dry = np.empty(NLayers)
    vsh_dry[:2] = (1.0 - phy["Poro1"]) * phy["vsh1"]
    vsh_dry[2:] = (1.0 - phy["Poro2"]) * phy["vsh2"]
    VSH = vsh_dry + WCont * chwt

    # heat capacity in intensity units
    dz = np.empty(NLayers)
    dz[0] = ZDpth[1] - ZDpth[0]
    dz[1:] = ZDpth[2:NLayers + 1] - ZDpth[0:NLayers - 1]
    HS = VSH * dz / (2.0 * DTSecs)

    # conductance divided by layer height
    GCond = np.empty(NLayers + 1)
    GCond[0] = BLCond
    GCond[1:] = CC / (ZDpth[1:NLayers + 1] - ZDpth[0:NLayers])
    return VSH, HS, GCond


def calc_cap_cond(CC, DyK, DyC, VSH, NLayers):
    """Helper coefficients for the profile update. Returns (condDZ, capDZ)."""
    condDZ = -(CC / DyK[:NLayers])
    capDZ = -(1.0 / (DyC[:NLayers] * VSH))
    return condDZ, capDZ


def calc_hstor(Tmp, TmpNw, HS0):
    """Heat stored to the surface from the previous step (calcHStor)."""
    T1Ave = (Tmp[1] + 3.0 * Tmp[2]) / 4.0
    TN1Ave = (TmpNw[1] + 3.0 * TmpNw[2]) / 4.0
    return HS0 * (TN1Ave - T1Ave)


def calc_profile(Tmp, condDZ, capDZ, NLayers, DTSecs, TrfFric, BLCond, RNet, LE_Flux):
    """One explicit time step of the ground temperature profile.
    Returns (TmpNw, GroundFlux)."""
    GFlux = np.empty(NLayers + 2)
    sensible = BLCond * (Tmp[0] - Tmp[1])
    GFlux[0] = RNet - LE_Flux + TrfFric + sensible
    GFlux[1:NLayers + 1] = condDZ * (Tmp[2:NLayers + 2] - Tmp[1:NLayers + 1])

    TmpNw = Tmp.copy()
    TmpNw[1:NLayers + 1] = (Tmp[1:NLayers + 1]
                            + DTSecs * capDZ * (GFlux[1:NLayers + 1] - GFlux[0:NLayers]))
    return TmpNw, GFlux[3]
