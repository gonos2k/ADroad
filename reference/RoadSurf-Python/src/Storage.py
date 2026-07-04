# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Functions for storage term calculation
"""
import Cond
#Determine predtype and add to surface water or snow layer
def PrecipitationToStorage(settings, CP, prec_phase, atm, surf):

    # Call the equivalent Python function to calculate precipitation type
    Cond.CalcPrecType(prec_phase, settings.DTSecs, atm, CP)

    # Update surface water and snow amounts based on precipitation type
    surf.SrfWatmms += atm.RainmmTS  # Rain (snow water not included)
    surf.SrfSnowmms += atm.SnowmmTS  # Snow precipitation (mm snow)
    
def melting(inCouplingPhase, TsurfObsLast, ground, surf, CP):
    """
    Calculate melting.

    Parameters:
    -----------
    inCouplingPhase : bool
        True if in the coupling phase.
    TsurfObsLast : float
        Last surface temperature observation.
    ground : GroundVariables
        Variables for ground properties.
    surf : SurfaceVariables
        Variables for surface properties.
    CP : RoadCondParameters
        Parameters to determine storage terms and road condition.

    Returns:
    --------
    None
    """
    QAvail = 0.0
    QLeftOver = 0.0
    
    # Check conditions for melting
    if (
        (surf.SrfSnowmms > 0.0 or surf.SrfIcemms > 0.0 or surf.SrfIce2mms > 0.0)
        and (
            ground.HStor > 0.00001
            and surf.TsurfAve > surf.T4Melt
            and surf.Q2Melt > 0
            and (not inCouplingPhase or TsurfObsLast < surf.T4Melt)
        )
    ):
        if not CP.CanMeltingChangeTemperature:
            return

        if (
            ground.HStor <= 0.00001
            or surf.TsurfAve <= surf.T4Melt
            or surf.Q2Melt <= 0
            or (inCouplingPhase and TsurfObsLast < surf.T4Melt)
        ):
            if surf.TsurfAve < 0.5:
                surf.Q2Melt = 0.0
                return
            elif surf.TsurfAve > 2.0:
                QAvail = ground.HS[0]*(ground.TmpNw[1] - surf.T4Melt) # Heat available for
                surf.Q2Melt = QAvail
                return

        # Heat available for melting
        QAvail = ground.HS[0] * (ground.TmpNw[1] - surf.T4Melt)

        # All available heat used (partly melts)
        if surf.Q2Melt >= QAvail:
            surf.Q2Melt = QAvail
            ground.TmpNw[1] = surf.T4Melt + 0.01  # Offset to guarantee freezing
            ground.TmpNw[2] = surf.T4Melt + 0.01
        else:
            # Only part used => no change in Q2Melt
            QLeftOver = QAvail - surf.Q2Melt
            ground.TmpNw[1] = surf.T4Melt + QLeftOver / ground.HS[0]  # Increases temperature
            ground.TmpNw[2] = surf.T4Melt + 0.01

       
        surf.TsurfAve = 0.5 * (ground.TmpNw[1] + ground.TmpNw[2])
    else:
        surf.Q2Melt = 0.0  # Set to zero value if no melting

def WaterStorage(MaxPormms, WatWear, surf, CP):
    """
    Update water storage parameters.

    Parameters:
    -----------
    MaxPormms : float
        Maximum water in asphalt pores.
    WatWear : float
        Water storage reduction caused by traffic.
    surf : SurfaceVariables
        Variables for surface properties.
    CP : RoadCondParameters
        Parameters to determine storage terms and road condition.

    Returns:
    --------
    SrfExtmms : float
        Surface water content.
    SrfPormms : float
        Water in pores.
    """
    SrfExtmms = 0.0

    # Evaporation / condensation
    # Only for bare and warm surface
    if (surf.SrfSnowmms <= 0.0 and surf.SrfIcemms <= 0.0 and
            surf.SrfDepmms <= 0.0 and surf.TsurfAve > CP.TLimDew):
        if surf.SrfWatmms > MaxPormms:
            # Surface evaporation/condensation
            surf.SrfWatmms -= surf.EvapmmTS
        else:
            # Pore evaporation/condensation
            surf.SrfWatmms -= CP.PorEvaF * surf.EvapmmTS
        
    # Water wear by traffic
    if surf.WearSurf and surf.SrfWatmms > 0.0:
        
        # Wear (only above threshold)
        if surf.SrfWatmms < CP.WWearLim:
            WatWear = 0.0  # No wear below specified limit
            
        if surf.SrfWatmms > CP.WWetLim:
            surf.SrfWatmms -= WatWear  # Water amount is reduced
           
        else:
            # Less wear below wet limit
            surf.SrfWatmms -= CP.DampWearF * WatWear
            
    # Water storage limits checked also at the end of storage calculations
    if surf.SrfWatmms < CP.MinWatmms:
        surf.SrfWatmms = 0.0  # Stop from going negative

    if surf.SrfWatmms > CP.MaxWatmms:
        surf.SrfWatmms = CP.MaxWatmms  # Overflow
        
def SnowStorage(Melted, DTSecs, wearF, MaxPormms, surf, CP, atm):
    """
    Calculate snow storage one time step forward.

    Parameters:
    -----------
    DTSecs : float
        Time step in seconds.
    wearF : WearingFactors
        Wearing factors.
    MaxPormms : float
        Maximum water content in pores.
    CP : RoadCondParameters
        Parameters to determine storage terms and road condition.
    surf : SurfaceVariables
        Variables for surface properties.
    atm : AtmVariables
        Variables for atmospheric properties.

    Returns:
    --------
    SrfPormms : float
        Water in pores.
    WatSnowRat : float
        Surface water to snow+water ratio.
    """
    SURFACE_SNOW_DRY = 1
    SURFACE_SNOW_WET = 2
    
    SrfExtmms = max((surf.SrfWatmms - MaxPormms), 0.0)  # Water on surface
    WatSnowRat = 0.0

    # Snow to snow+ice ratio
    RDummy = surf.SrfSnowmms + surf.SrfIcemms
    if RDummy > 0.001:
        CP.SnowIceRat = surf.SrfSnowmms / RDummy
    else:
        CP.SnowIceRat = 0.0  # Interpret surface icy

    if surf.SrfSnowmms > 0.0:
        # Wet snow: forming (initially dry)
        WatSnowRat = SrfExtmms / (SrfExtmms + surf.SrfSnowmms)
        if WatSnowRat > CP.WetSnowFormR:
            atm.SnowType = SURFACE_SNOW_WET
    else:
        # Only dry => wet allowed
        atm.SnowType = SURFACE_SNOW_DRY

    if surf.SrfSnowmms > 0.0:
        # Deposit under snow
        if surf.SrfDepmms > 0.0:
            # ... to ice
            surf.SrfIcemms += surf.SrfDepmms
            # Deposit storage to zero
            surf.SrfDepmms = 0.0

    if surf.SrfSnowmms > 0.0:
        # Force melting when salt on road
        if CP.forceSnowMelting:
            surf.SrfWatmms += surf.SrfSnowmms
            surf.SrfSnowmms = 0.0
        # Normal melting
        elif surf.Q2Melt > 0.0 and surf.TsurfAve >= CP.TLimMeltSnow:
            # Melted amount (meters/timestep)
            melted = (surf.Q2Melt * DTSecs) / (CP.WatMHeat * CP.WatDens)
            surf.SrfSnowmms -= 1000.0 * melted  # Snow melts
            surf.SrfWatmms += 1000.0 * melted  # Forms water

    if surf.WearSurf and surf.SrfSnowmms > 0.0:
        # Wear: snow wears to ice
        # Snow amount is reduced to ice
        surf.SrfSnowmms -= wearF.SnowTran
        surf.SrfIcemms += CP.Snow2IceFac * wearF.SnowTran
        surf.SrfIce2mms += CP.Snow2IceFac * wearF.SnowTran
    
    # Wet snow
    if surf.SrfSnowmms > 0.0 and atm.SnowType == SURFACE_SNOW_WET:
        if WatSnowRat > CP.WetSnowMeltR:
            # Melting (high water content)
            surf.SrfWatmms += surf.SrfSnowmms  # Forms water
            surf.SrfSnowmms = 0.0  # Snow melts
            atm.SnowType = SURFACE_SNOW_DRY  # Default snow type
    
        if surf.TsurfAve < CP.TLimFreeze:
            # Freezing - all at once to ice
            surf.SrfIcemms += surf.SrfSnowmms + surf.SrfWatmms  # Including water storage
            surf.SrfIce2mms += surf.SrfSnowmms + surf.SrfWatmms
            atm.SnowType = SURFACE_SNOW_DRY  # Default snow type
            if surf.SrfSnowmms > 0.5:
                CP.WetSnowFrozen = True  # Wet snow is frozen
            surf.SrfSnowmms = 0.0
            surf.SrfWatmms = 0.0
    
    if surf.SrfSnowmms < CP.MinSnowmms:
        surf.SrfSnowmms = 0.0  # Stop from going too small
    
    if surf.SrfSnowmms > CP.MaxSnowmms:
        # Snow "overflow" - reduce snow amount to half
        surf.SrfSnowmms -= CP.MaxSnowmms / 2.0


def IceStorage(Melted, DTSecs, surf, CP, wearF):
    # *************  ICE STORAGE
    # Heat needed for melting is calculated at the end of the subroutine,
    # and melting heat balance in RoadTemp
    
    if surf.TsurfAve < CP.TLimFreeze and surf.SrfWatmms > 0.0:  # Freezing
        surf.SrfIcemms += surf.SrfWatmms  # All water to ice
        surf.SrfIce2mms += surf.SrfWatmms  # No freezing for deposit
        surf.SrfWatmms = 0.0

    if surf.SrfSnowmms <= 0.0 and surf.SrfIcemms > 0.0:  # Melting
        if CP.forceIceMelting:
            surf.SrfWatmms += surf.SrfIcemms
            surf.SrfIcemms = 0.0
            surf.SrfIce2mms = 0.0
        # Only on snow-free ice
        elif surf.Q2Melt > 0.0 and surf.TsurfAve >= CP.TLimMeltIce:
            # Melted amount (meters/timestep)
            Melted = (surf.Q2Melt * DTSecs) / (CP.WatMHeat * CP.WatDens)

            surf.SrfIcemms -= 1000.0 * Melted  # Both at the same rate
            surf.SrfIce2mms -= 1000.0 * Melted
            surf.SrfWatmms += 1000.0 * Melted  # Adds water storage

    if surf.WearSurf and surf.SrfIcemms > 0.0:  # Wear: also under snow
        # Ice amount is reduced to secondary ice at a faster rate
        surf.SrfIcemms -= wearF.IceWear

    if surf.WearSurf and surf.SrfIce2mms > 0.0:  # Wear: also under snow
        # Ice amount is reduced to secondary ice at a faster rate
        surf.SrfIce2mms -= wearF.IceWear2

    if surf.SrfIcemms < CP.MinIcemms:
        surf.SrfIcemms = 0.0  # Stop from going too small

    if surf.SrfIcemms > CP.MaxIcemms:  # Ice "overflow"
        surf.SrfIcemms = CP.MaxIcemms  # Reduce to maximum

    if surf.SrfIce2mms < CP.MinIcemms:
        surf.SrfIce2mms = 0.0  # Stop from going too small

    if surf.SrfIce2mms > CP.MaxIcemms:  # Ice "overflow"
        surf.SrfIce2mms = CP.MaxIcemms  # Reduce to maximum
   

def DepositStorage(DepWear, surf, CP):
    # *************  DEPOSIT STORAGE
    # Removed conditions for ice-free surface and TSurf <= CP%TLimDew
   
    if surf.EvapmmTS < 0.0:  # Condensation only
        surf.SrfDepmms -= surf.EvapmmTS  # No evaporation

    if surf.TsurfAve > CP.TLimMeltDep:  # Melting
        
        surf.SrfWatmms += surf.SrfDepmms  # Increase water storage
        surf.SrfDepmms = 0.0  # All deposit to water  

    if surf.WearSurf and (surf.SrfSnowmms <= 0.0) and (surf.SrfDepmms > 0):  # Wear only on snow-free surface
        surf.SrfDepmms -= DepWear
        
    if surf.SrfDepmms < CP.MinDepmms:
        surf.SrfDepmms = 0.0  # Stop from going too small

    if surf.SrfDepmms > CP.MaxDepmms:  # Deposit "overflow"?
        # Excess into water storage
        surf.SrfWatmms += surf.SrfDepmms - CP.MaxDepmms
        surf.SrfDepmms = CP.MaxDepmms  # Reduce to maximum
    
def NewMeltFreezeHeat(DTSecs, surf, CP):
    # Calculate the heat needed to melt/freeze the whole uppermost snow/ice layer.
    # Used and updated in the energy balance calculation.
    # Thicknesses in equivalent water mm.

    # Melting for snow and ice
    surf.Q2Melt = 0.0  # Default

    if surf.SrfSnowmms > 0.0:
        # Snow melt heat (W/m2)
        surf.Q2Melt = (CP.WatMHeat * CP.WatDens * (surf.SrfSnowmms / 1000.0) / DTSecs)
        surf.T4Melt = CP.TLimMeltSnow

    if surf.SrfSnowmms <= 0.0 and surf.SrfIcemms > 0.0:
        # Ice melt heat (W/m2)
        surf.Q2Melt = (CP.WatMHeat * CP.WatDens * (surf.SrfIcemms / 1000.0) / DTSecs)
        surf.T4Melt = CP.TLimMeltIce

    if surf.Q2Melt < 0.0:
        surf.Q2Melt = 0.0  # Just to be sure...

