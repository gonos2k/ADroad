# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Functions related to heat balance calculation
"""
import BoundaryLayer
import Storage
import numpy as np
from datetime import datetime

def calculate_julian_day():
    # Calculate the Julian day based on the current date
    today = datetime.now()
    jan1 = datetime(today.year, 1, 1)
    juld = (today - jan1).days + 1
    return juld

def CalcHCapHCond(NLayers, DTSecs, phy, ground, atm):
    """
    Calculate ground layers' heat capacity and heat conductance.

    Parameters:
    -----------
    NLayers : int
        Number of ground layers.
    DTSecs : float
        Time step in seconds.
    phy : PhysicalParameters
        Physical parameters used in the model.
    ground : GroundVariables
        Variables for ground properties.
    atm : AtmVariables
        Variables for atmospheric properties.

    Returns:
    --------
    None
    """
    ground.GCond[0] = atm.BLCond  # Boundary layer conductance
    
    for i in range(0, NLayers):
        tmp2 = ground.TmpNw[i+1] ** 2
        if ground.TmpNw[i+1] >= 0:  # Water
            # Water density (kg/m3)
            RooWT = -0.0050 * tmp2 + 0.0079 * ground.TmpNw[i+1] + 1000.0028
            # Specific heat capacity of water (kJ/kgK)
            CWT = 0.0000102 * tmp2*tmp2 - 0.0017169 * tmp2 * ground.TmpNw[i+1] + 0.11516 * tmp2 - 3.4739 * ground.TmpNw[i+1] + 4217.2
            
        else:  # Ice
            RooWT = 920.0
            CWT = 2100.0

        CHWT = RooWT * CWT  # Volumetric heat capacity for water
        
        if i <= 1:
            ground.VSH[i] = (1.0 - phy.Poro1) * phy.vsh1 + ground.WCont[i] * CHWT
            
        else:
           
            ground.VSH[i] = (1.0 - phy.Poro2) * phy.vsh2 + ground.WCont[i] * CHWT
        
        
        if i == 0:
            ground.HS[i] = ground.VSH[i] * (ground.ZDpth[i + 1] - ground.ZDpth[i]) / (2.0 * DTSecs)
        else:
            ground.HS[i] = ground.VSH[i] * (ground.ZDpth[i + 1] - ground.ZDpth[i - 1]) / (2.0 * DTSecs)

        
        ground.GCond[i+1] = ground.CC[i] / (ground.ZDpth[i + 1] - ground.ZDpth[i])

def calcCapDZCondDZ(NLayers, ground):
# Calculate help variables to use in the temperature profile calculation
    ground.condDZ[0] = -(ground.CC[0] / ground.DyK[0])
    ground.capDZ[0] = -(1 / (ground.DyC[0] * ground.VSH[0]))
    
    for j in range(1, NLayers):
        ground.condDZ[j] = -(ground.CC[j] / ground.DyK[j])
        ground.capDZ[j] = -(1 / (ground.DyC[j] * ground.VSH[j]))
        
def BalanceModelOneStep(SWi, LWi, phy, ground, surf, atm, settings, coupling, modelInput, inputIdx, condParam):
    """
    Subroutine to perform one step of the balance model.

    Parameters:
    -----------
    SWi : float
        Downwelling shortwave radiation.
    LWi : float
        Downwelling longwave radiation.
    phy : PhysicalParameters
        Physical parameters used in the model.
    ground : GroundVariables
        Variables for ground properties.
    surf : SurfaceVariables
        Variables for surface properties.
    atm : AtmVariables
        Variables for atmospheric properties.
    settings : ModelSettings
        Variables for model settings.
    coupling : CouplingVariables
        Variables used in coupling.
    modelInput : inputArrays
        Arrays for model input data.
    inputIdx : int
        Index in input data.
    condParam : RoadCondParameters
        Parameters to determine storage terms and road condition.
    """
    # Set traffic friction and check that wind is not below the minimum
    SetDayDependendVariables(settings, surf, modelInput, atm, inputIdx)
    # Calculate boundary layer conductance
    BoundaryLayer.CalcBLCondAndLE(surf, settings.DTSecs, surf.SrfWatmms, phy, atm)
    # Calculate net radiation
    atm.RNet=CalcRNet(phy.Emiss, phy.SB_const, surf, ground.Albedo, SWi, LWi,
             coupling.SWRadCof, coupling.LWRadCof)

    # Calculate heat capacity and conductance
    CalcHCapHCond(settings.NLayers, settings.DTSecs, phy, ground, atm)

    # Calculate heat capacity and conductance derivatives
    calcCapDZCondDZ(settings.NLayers, ground)

    # Calculate temperature profile for the next time step
    calcProfile(settings.NLayers, settings.DTSecs, surf.TrfFric, ground, atm)

    # Calculate heat storage
    calcHStor(ground)

    # Check if melting
    Storage.melting(coupling.inCouplingPhase, coupling.LastTsurfObs, ground, surf, condParam)

    ground.Tmp = ground.TmpNw.copy()


    # Otherwise, use the average of the first two layers
    surf.TsurfAve = (ground.Tmp[1] + ground.Tmp[2]) / 2.0
    

def SetDayDependendVariables(settings, surf, modelInput, atm, inputIdx):
    """
    Set traffic friction and check that wind is not below the minimum.

    Parameters:
    -----------
    settings : ModelSettings
        Variables for model settings.
    surf : SurfaceVariables
        Variables for surface properties.
    modelInput : inputArrays
        Model input arrays.
    atm : AtmVariables
        Variables for atmospheric properties.
    inputIdx : int
        Input data index.
    """
    shour = modelInput.time[inputIdx].hour

    if (shour >= settings.NightOn) or (shour <= settings.NightOff):
        # Night time
        # Emulate heat effect of traffic by wind-induced turbulence and friction
        # Different coefficients for day and night
        atm.CalmLim = settings.CalmLimNgt
        surf.TrfFric = settings.TrfFricNgt
    else:
        # Daytime
        atm.CalmLim = settings.CalmLimDay
        surf.TrfFric = settings.TrfFricDay

    # Minimum wind speed
    if atm.VZ < atm.CalmLim:
        atm.VZ = atm.CalmLim



def CalcRNet(Emiss, SB_Const, surf, Albedo, SW, LW, SwRadCof, LWRadCof):
    """
    Calculate net radiation.

    Parameters:
    -----------
    Emiss : float
        Emissivity constant of the surface.
    SB_Const : float
        Stefan-boltzman constant (W/m2K4).
    TsurfAve : float
        Average temperature of the first two layers.
    Albedo : float
        Surface albedo.
    SW : float
        Downwelling short wave radiation.
    LW : float
        Downwelling long wave radiation.
    RNet : float
        Net radiation.
    SwRadCof : float
        Radiation coefficient to use for short wave radiation.
    LWRadCof : float
        Radiation coefficient to use for long wave radiation.
    """
    TsurfK = surf.TsurfAve + 273.15
    TsurfK2 = TsurfK * TsurfK
    RBB = Emiss * SB_Const * (TsurfK2 * TsurfK2)  # Black body emission
    RNet = (1.0 - Albedo) * SW * SwRadCof + Emiss * LW * LWRadCof - RBB
    
    return RNet


def calcHStor(ground):
    """
    Calculate heat variable describing the stored heat to the surface from the previous time step.

    Parameters:
    -----------
    ground : GroundVariables
        Ground variables.

    Returns:
    --------
    None
    """
    T1Ave = (ground.Tmp[1] + 3. * ground.Tmp[2]) / 4.
    TN1Ave = (ground.TmpNw[1] + 3. * ground.TmpNw[2]) / 4.

    ground.HStor = ground.HS[0] * (TN1Ave - T1Ave)

def calcProfile(NLayers, DTSecs, TrfFric, ground, atm):
    """
    Calculate temperature profile one time step forward.

    Parameters:
    -----------
    NLayers : int
        Number of ground layers.
    DTSecs : float
        Time step in seconds.
    TrfFric : float
        Surface heating caused by traffic.
    ground : GroundVariables
        Variables for ground properties.
    atm : AtmVariables
        Variables for atmospheric properties.

    Returns:
    --------
    None
    """
    GFlux = np.zeros(NLayers + 2)  # Heat flux

    atm.SensibleHeatFlux = atm.BLCond * (ground.Tmp[0] - ground.Tmp[1])
    
    # Heat flux from air to ground
    GFlux[0] = atm.RNet - atm.LE_Flux + TrfFric + atm.SensibleHeatFlux
    
    ground.TmpNw = ground.Tmp.copy()

    # Calculate heat flux for different layers
    for j in range(1, NLayers + 1):
        GFlux[j] = ground.condDZ[j-1] * (ground.Tmp[j + 1] - ground.Tmp[j])

    ground.GroundFlux = GFlux[3]

    # Calculate new temperatures
    
    for j in range(1, NLayers + 1):
        ground.TmpNw[j] = ground.Tmp[j] + DTSecs * (ground.capDZ[j-1] * (GFlux[j] - GFlux[j - 1]))

