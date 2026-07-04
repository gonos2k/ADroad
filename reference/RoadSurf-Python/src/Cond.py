# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Functions for calculating changes in stroage terms (snow, water, ice ,deposit)
All storage terms are in water equivalen mm
"""
import math
import Storage

def CalcPrecType(PrecPhase, DTSecs, atm, CP):
    #         ************* PRECIPTATION TYPE
    # * Snow amount set by in-built interpretation if left to missing value or if forced
    #   - Ref: Koistinen, Saltikoff: Experience on customer products of accumulated snow,
    #  sleet and rain(Int.Seminar on Advanced Weather Radar Systems, Switzerland, March 1998)
    # * atm%PrecType set to (-1 = not defined; should appear onply if PrecInTStep <MinPrecmm)
    #    1 = water
    #    2 = sleet
    #    3 = snow
    
       atm.RainmmTS = 0.0
       atm.SnowmmTS = 0.0
       atm.PrecType = -1 # Reset atm.PrecType
       UseInterpr = True # Reset in-built phase interpretation control
       
       PRECIPITATION_NONE = 0
       PRECIPITATION_RAIN = 1
       PRECIPITATION_SLEET = 2
       PRECIPITATION_SNOW = 3
       PRECIPITATION_HAIL = 6
       PRECIPITATION_FREEZING_DRIZZLE = 4
       PRECIPITATION_FREEZING_RAIN = 5
       SURFACE_SNOW_WET = 2
       if PrecPhase > CP.MissValI:
           UseInterpr = False  # By default, no interpretation
        
           if atm.PrecInTStep <= CP.MinPrecmm:
                atm.PrecInTStep = 0.0  # Check minimum level
                atm.PrecType = -1  # No precipitation -> No PrecType
                atm.RainmmTS = 0.0
                atm.SnowmmTS = 0.0
           else:
                if PrecPhase > CP.MissValI:
                    # WATER (above minimum)
                    if PrecPhase in [PRECIPITATION_NONE, PRECIPITATION_RAIN, 
                                       PRECIPITATION_FREEZING_DRIZZLE, PRECIPITATION_FREEZING_RAIN]:
                        atm.RainmmTS = atm.PrecInTStep  # Interpretation should handle freezing drizzle and rain
                        atm.SnowmmTS = 0.0  # Rain on snow makes it wet
                        atm.PrecType = 1  #water
                        atm.SnowType = SURFACE_SNOW_WET  # Wet snow
                    elif PrecPhase == PRECIPITATION_SLEET:  # SLEET
                        atm.SnowmmTS = atm.PrecInTStep / 2.0  # Half water, half snow
                        atm.RainmmTS = atm.SnowmmTS  # Wet snow
                        atm.PrecType = 2  #sleet
                        atm.SnowType = SURFACE_SNOW_WET
                    elif PrecPhase in [PRECIPITATION_SNOW, PRECIPITATION_HAIL]:  # SNOW
                        atm.SnowmmTS = atm.PrecInTStep  # All precipitation as snow
                        atm.PrecType = 3  #snow
                        atm.RainmmTS = 0.0
                    else:  # INTERPRETATION MISSING
                        UseInterpr = True  # Force own interpretation


       if UseInterpr:
           if atm.PrecInTStep <= CP.MinPrecmm:
                atm.PrecInTStep = 0.0  # Check minimum level
                atm.PrecType = -1  # No precipitation -> No PrecType
                atm.RainmmTS = 0.0
                atm.SnowmmTS = 0.0
           else:
                atm.SnowmmTS = 0.0  # Amount of snow (mm_water/time_step)
                p_exp = 22.0 - 2.7 * atm.Tair - 0.20 * atm.Rhz
                p_rain = 1.0 / (1.0 + math.exp(p_exp))
        
                if p_rain < CP.PLimSnow:  # SNOW
                    atm.SnowmmTS = atm.PrecInTStep
                    atm.PrecType = 3
                elif p_rain > CP.PLimRain:  # WATER
                    atm.RainmmTS = atm.PrecInTStep
                    atm.SnowType = SURFACE_SNOW_WET  # Rain on snow makes it wet
                    atm.PrecType = 1
                else:  # SLEET
                    atm.SnowmmTS = atm.PrecInTStep / 2.0  # Half water, half snow
                    atm.RainmmTS = atm.SnowmmTS
                    atm.SnowType = SURFACE_SNOW_WET
                    atm.PrecType = 2

       atm.RainIntensity = (atm.RainmmTS / DTSecs) * 3600.0  # Intensity, rain mm/h
       atm.SnowIntensity = (atm.SnowmmTS / DTSecs) * 3600.0  # Intensity, snow mm/h

#Storage wear by traffic
def WearFactors(condParam, Tph, surf, wearF):
    """
    Calculate wear factors.

    Parameters:
    -----------
    Snow2IceFac : float
        Snow to ice transition factor.
    Tph : float
        Time steps per hour.
    surf : SurfaceVariables
        Variables for surface properties.

    Returns:
    --------
    wearF : WearingFactors
        Calculated wearing factors.
    """

    # Calculate wear factors
    wearF.SnowTran = (0.2 + 0.25) * surf.SrfSnowmms
    wearF.SnowTran = max(wearF.SnowTran, 0.01)

    # In case of a small snow layer (< 0.2mm), wearing more effective (x3)
    if surf.SrfSnowmms < 0.2:
        wearF.SnowTran *= 3

    condParam.Snow2IceFac = 0.25 / (0.2 + 0.25)
    wearF.SnowTran *= Tph  # to mm/time step

    wearF.IceWear = 1.1 * 2.0 * 0.145 * surf.SrfIcemms  # car traffic
    wearF.IceWear = max(wearF.IceWear, 0.01)
    wearF.IceWear *= Tph  # to mm/time step

    wearF.IceWear2 = 1.1 * 2.0 * (4.0 * 0.290) * surf.SrfIce2mms  # car traffic
    wearF.IceWear2 = max(wearF.IceWear2, 0.01)
    wearF.IceWear2 *= Tph  # to mm/time step

    wearF.DepWear = 0.5 * 2.0 * (4.0 * 0.290) * surf.SrfDepmms  # car traffic
    wearF.DepWear = max(wearF.DepWear, 0.01)
    wearF.DepWear *= Tph  # to mm/time step

    wearF.WatWear = 0.145 * surf.SrfWatmms  # car traffic
    wearF.WatWear = max(wearF.WatWear, 0.06)
    wearF.WatWear = 10 * wearF.WatWear * Tph  # to mm/time step

    return wearF

def RoadCond(MaxPormms, surf, atm, settings, CP, wearF):
    """
    Update road condition parameters.

    Parameters:
    -----------
    MaxPormms : float
        Maximum water in asphalt pores.
    surf : SurfaceVariables
        Variables for surface properties.
    atm : AtmVariables
        Variables for atmospheric properties.
    settings : ModelSettings
        Variables for model settings.
    CP : RoadCondParameters
        Parameters to determine storage terms and road condition.
    wearF : WearingFactors
        Wearing factors.

    Returns:
    --------
    None
    """
    Melted = 0.0
    CP.SnowIceRat = 0.0  # Snow to snow+ice ratio
    SURFACE_SNOW_DRY=1
    atm.SnowType = SURFACE_SNOW_DRY

    # Between the limits, keep the old setting hysteresis: *H > *L
    if surf.VeryCold and surf.TsurfAve > CP.TLimColdH:
        surf.VeryCold = False

    if not surf.VeryCold and surf.TsurfAve < CP.TLimColdL:
        surf.VeryCold = True  
    # Water Storage
    Storage.WaterStorage(MaxPormms, wearF.WatWear, surf,CP)
    # Snow Storage
    Storage.SnowStorage(Melted, settings.DTSecs, wearF,MaxPormms, surf,CP,atm)  
    # Ice Storage
    Storage.IceStorage(Melted, settings.DTSecs, surf, CP, wearF)
    # Deposit Storage
    Storage.DepositStorage(wearF.DepWear, surf, CP)   
    # Water Storage Limits Recheck
    if surf.SrfWatmms < CP.MinWatmms:
        surf.SrfWatmms = 0.0  # Stop from going negative

    if surf.SrfWatmms > CP.MaxWatmms:
        surf.SrfWatmms = CP.MaxWatmms  # Overflow

    Storage.NewMeltFreezeHeat(settings.DTSecs, surf, CP)

 #Calculate surface alabedo  
def CalcAlbedo(surf, CP):
    # Variables for surface properties
    Albedo = None
    IceSum = 0.0
    IceMax = 0.0

    # Surface condition dependent
    if surf.WearSurf:
        IceSum = 0.5 * (surf.SrfIcemms + surf.SrfIce2mms) + surf.SrfDepmms
        if IceSum < 0.0:
            IceSum = 0.0
        IceMax = 1.5
        Albedo = CP.AlbDry

        if surf.SrfSnowmms > 0.01 and surf.SrfSnowmms > surf.SrfIcemms:
            Albedo = CP.AlbSnow
        elif surf.SrfIcemms > 0.01 or surf.SrfDepmms > 0.01:
            if IceSum < IceMax:
                Albedo = CP.AlbDry + (IceSum / IceMax) * (CP.AlbSnow - CP.AlbDry)
            else:
                Albedo = CP.AlbSnow

    return Albedo
