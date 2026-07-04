# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Functions for calculating boundary layer conductance, evaporation and
latent heat flux
"""
import numpy as np
from RoadSurfVariables import*

def CalcBLCondAndLE(surf, DtSecs, SrfWatmms, phy, atm):              
    Tair = atm.Tair
    VZ = atm.VZ
    Rhz = atm.Rhz
    BLCond = atm.BLCond
    
    TaK = Tair + 273.15
    ConvLim = 0.001
    
    AirDens = 100000.0 / (287.05 * TaK)
    AirHCap = 1005.0 + ((TaK - 250.0) ** 2) / 3364.0
    AirVCap = AirHCap * AirDens
    
    WatDen = -0.0050 * surf.TsurfAve ** 2 + 0.0079 * surf.TsurfAve + 1000.0028
    
    PSIM = 0.0 # STAB. CORR. FACTOR FOR MOMENTUM (INITIAL VALUE)
    PSIH = 0.0 # STAB. CORR. FACTOR FOR HEAT (INITIAL VALUE)
    Stab= 0.0
    MaxIter=40
    for j in range(1, MaxIter + 1):
        
        BLCond_Old = BLCond
    # FRICTION VELOCITY
        UStar = phy.VK_Const * VZ / (phy.logUstar + PSIM)
    
        if UStar < 0.0:
            print("ERROR: UStar is negative")
            print(f"Tair: {Tair}, VZ: {VZ}, Rhz: {Rhz}, BLCond: {BLCond}, TSurfAve: {surf.TSurfAve}")
    
        # BOUNDARY LAYER CONDUCTANCE / STABILITY PARAMETER
        BLCond = AirVCap * phy.VK_Const * UStar / (phy.logCond + PSIH)
        Stab = -phy.VK_Const * phy.ZRefT * phy.Grav * BLCond * (surf.TsurfAve - Tair) / (
                    AirVCap * (Tair + 273.15) * (UStar ** 3))
        if Stab > 1:
            Stab = 1
    
        # STABILITY CORRECTION FACTORS
        if Stab > 0:  # STABLE CONDITION
            PSIH = 4.7 * Stab
            PSIM = PSIH
        else:  # UNSTABLE CONDITION
            PSIH = -2.0 * np.log((1.0 + np.sqrt(1.0 - 16.0 * Stab)) / 2.0)
            PSIM = 0.6 * PSIH
    
        # CHECK ITERATION CONVERGENCE
        if abs(BLCond - BLCond_Old) < ConvLim and j >= 5:
            break
    
        BLCond_Old = BLCond

    if abs(BLCond - BLCond_Old) > 10 * ConvLim and j >= 5:
       print(f"Max number of BLCond iterations (MaxIter, BLCond_Old, BLCond): {j}, {BLCond_Old}, {BLCond}")

    Raero = calcRaero(phy.logMom, phy.logHeat, PSIM, PSIH, phy.VK_Const, VZ)

    atm.LE_Flux, surf.EvapmmTS = CalcLE(surf.TsurfAve, Tair, Rhz, AirDens, AirHCap, 
                                        Raero, phy.LVap, phy.LFus, WatDen, DtSecs, SrfWatmms)
    
    atm.BLCond = BLCond  # Set BLCond to the result
   

def calcRaero(logMom, logHeat, PSIM, PSIH, VK_Const, VZ):
    RAero = (logMom + PSIM) * (logHeat + PSIH) / (VK_Const ** 2 * VZ)
    if RAero > 30.0:
        RAero = 30.0
    return RAero

def CalcLE(TSurfAve, TAmb, Rhz, AirDens, AirHCap, RAero, LVap, LFus, WatDen, DtSecs, SrfWatmms):
    TsurfAve = TSurfAve
    ESat = 0.0
    ESurf = 0.0
    EAir = 0.0
    TaK = TAmb + 273.15
    PsychC = 0.1 * (0.00063 * TaK + 0.47496)

    if TsurfAve < 0:
        ESat = 0.61078 * np.exp(21.875 * TsurfAve / (TsurfAve + 265.5))
    else:
        ESat = 0.61078 * np.exp(17.269 * TsurfAve / (TsurfAve + 237.3))
    ESurf = ESat

    if TAmb < 0:
        ESat = 0.61078 * np.exp(21.875 * TAmb / (TAmb + 265.5))
    else:
        ESat = 0.61078 * np.exp(17.269 * TAmb / (TAmb + 237.3))
    EAir = min((0.01 * Rhz), 1.0) * ESat

    LE_Flux = (AirDens * AirHCap * (ESurf - EAir)) / (PsychC * RAero)

    if TsurfAve >= 0.0:
        EvapmmTS = (LE_Flux / (LVap * WatDen)) * 1000.0 * DtSecs
    else:
        EvapmmTS = (LE_Flux / (LFus * WatDen)) * 1000.0 * DtSecs

    if (LE_Flux > 0.0) and (SrfWatmms <= 0.0):
        LE_Flux = 0.0
        EvapmmTS = 0.0
    
    return LE_Flux, EvapmmTS
