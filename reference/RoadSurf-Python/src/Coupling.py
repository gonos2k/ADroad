# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Functions related to coupling
Coupling determines a radiation correction coeffiecient to make modeled road surface
temperature to fit observations at the end of the initialization phase
"""
import numpy as np
from RoadSurfVariables import*

def CouplingOperations1(i, coupling, surf, settings, ground, modelInput, CP, localParam):
    DTs = settings.DTSecs

    coupling.inCouplingPhase = False
    # Determine if simulation in coupling phase
    if i >= coupling.couplingStartI and i <= coupling.couplingEndI:
        coupling.inCouplingPhase = True

    # At start of coupling period save simulated values so that they can be restored later
    if i == coupling.couplingStartI and coupling.Coupling_iterations == 0:
        saveDataForCoupling(i, ground, coupling, surf, modelInput, settings)
        # Initialize coefficients
        coupling.SwRadCof = 1.0
        coupling.LWRadCof = 1.0
        coupling.SW_correction = 0.0
        coupling.LW_correction = 0.0

    # If coupling is started again, return to start of coupling period
    if coupling.start_coupling_again:
        # Restore parameters at the beginning of coupling
        i=uploadDataForCoupling(i, ground, coupling, surf, modelInput, settings)
        coupling.start_coupling_again = False
        # Set coefficient for short wave radiation if it has larger value than Long wave radiation or sky view factor is NOT used
        if modelInput.SW[i] > modelInput.LW[i] and not (localParam.sky_view < 1.0 and localParam.sky_view > -0.01):
            coupling.SwRadCof = coupling.RadCoeff
            coupling.LWRadCof = 1.0
        else:
            # Otherwise set coefficients for long wave radiation
            coupling.SwRadCof = 1.0
            coupling.LWRadCof = coupling.RadCoeff

    # Radiation coefficients after coupling
    # Returns gradually to 1
    if i > coupling.couplingEndI:
        coupling.SwRadCof = 1.0 + coupling.SW_correction * np.exp(-((DTs * i) \
            - (DTs * coupling.couplingEndI)) / settings.couplingEffectReduction)
        coupling.LWRadCof = 1.0 + coupling.LW_correction * np.exp(-((DTs * i) \
            - (DTs * coupling.couplingEndI)) / settings.couplingEffectReduction)
    
    if coupling.inCouplingPhase:
        # Check if there is a need to force snow/ice melting
        # to prevent coupling from getting stuck
        snowIceCheck(coupling.LastTsurfObs, surf, CP)
    return i

def saveDataForCoupling(datai, ground, coupling, surf, model_input, settings):
    coupling_len = coupling.couplingEndI - coupling.couplingStartI + 1

    coupling.saveDatai = datai
    coupling.TsurfAveSave = surf.TsurfAve
    coupling.srfWatmmsSave = surf.SrfWatmms
    coupling.srfIce2mmsSave = surf.SrfIce2mms
    coupling.srfIce2mmsSave = surf.SrfIce2mms
    coupling.srfDepmmsSave = surf.SrfDepmms
    coupling.srfSnowmmsSave = surf.SrfSnowmms
    coupling.AlbedoSave = ground.Albedo
    coupling.VeryColdSave = surf.VeryCold

    coupling.TmpSave = ground.Tmp.copy()

    for i in range(1, coupling_len + 1):
        coupling.SWSave=model_input.SW[datai]
        coupling.SWDirSave=model_input.SW_dir[datai]
        coupling.LWSave=model_input.LW[datai]
        
def uploadDataForCoupling(datai, ground, coupling, surf, modelInput, settings):
    couplingLen = coupling.couplingEndI - coupling.couplingStartI + 1
    datai = coupling.saveDatai
    surf.TsurfAve = coupling.TsurfAveSave
    surf.srfWatmms = coupling.SrfWatmmsSave
    surf.srfIce2mms = coupling.SrfIce2mmsSave
    surf.srfDepmms = coupling.SrfDepmmsSave
    surf.srfSnowmms = coupling.SrfSnowmmsSave
    ground.Albedo = coupling.AlbedoSave
    surf.VeryCold = coupling.VeryColdSave

    
    ground.Tmp = coupling.TmpSave.copy()

    #for i in range(1, couplingLen + 1):
    #    modelInput.SW[coupling.couplingStartI + i - 1] = coupling.SWSave
    #    modelInput.SW_dir[coupling.couplingStartI + i - 1] = coupling.SWDirSave
    #    modelInput.LW[coupling.couplingStartI + i - 1] = coupling.LWSave

    return datai

def snowIceCheck(LastTsurfObs, surf, CP):
    if LastTsurfObs > CP.TLimMeltSnow and surf.SrfSnowmms > 0.00:
        # Force snow melting
        surf.SrfWatmms += surf.SrfSnowmms
        surf.SrfSnowmms = 0.00

    if LastTsurfObs > CP.TLimMeltIce and surf.SrfIcemms > 0.00:
        # Force ice melting
        surf.SrfWatmms += surf.SrfIcemms
        surf.SrfIcemms = 0.00

    if LastTsurfObs > CP.TLimMeltIce and surf.SrfIce2mms > 0.00:
        # Force ice melting
        surf.SrfWatmms += surf.SrfIce2mms
        surf.SrfIce2mms = 0.00

    if LastTsurfObs > CP.TLimMeltDep and surf.SrfDepmms > 0.00:
        # Force ice melting
        surf.SrfWatmms += surf.SrfDepmms
        surf.SrfDepmms = 0.00

def CheckEndCoupling(i, settings, coupling, surf):
    # Variables for model settings, surface properties, and coupling
 
    # Coupling control if at the end of the coupling period
    if settings.use_coupling and i == coupling.couplingEndI and not coupling.coupling_failed:
        # Determine new radiation coefficient if necessary
        CouplingOperations2(surf, coupling)
        
def CouplingOperations2(surf, coupling):

    # Save temperature if at the first iteration
    if coupling.Coupling_iterations == 0:
        coupling.Tsurf_end_coup1 = surf.TsurfAve

    # Change radiation coefficient if necessary
    Coupling_control(surf.TsurfAve, coupling)

    # Increment coupling iterations
    coupling.Coupling_iterations += 1
    
def Coupling_control(TsurfAve, coupling):
    # Constants
    TDifBelow = 0.0  # Initialize to 0.0
    TDifAbove = 0.0  # Initialize to 0.0

    #Crevier, L. and Y. Delage, 2001: METRo: A new model for road-condition forecasting
    #in Canada. Journal of Applied Meteorology, 40(11), 2026-2037
    
    #Karsisto, V. P. Nurmi, M. Kangas, M. Hippi, C. Fortelius, S. Niemelä and 
    #H. Järvinen, 2016: Improving road weather model forecasts by adjusting the radiation
    #input. Meteorological Applications, 23, 503-513
    
    coupling.start_coupling_again = False

    # Change to Kelvins
    TsurfAve += 273.16
    coupling.LastTsurfObs += 273.16

    # If coupling is not failed
    if not coupling.coupling_failed:

        if coupling.Coupling_iterations == 0:
            coupling.Tsurf_end_coup1 = TsurfAve  # Save first value (radcof=1)

        # If too many iterations
        if coupling.Coupling_iterations == 25:

            # If first guess was better than last, return to start and use radcof=1
            # Otherwise, continue from here with radcof=1
            if abs(coupling.Tsurf_end_coup1 - coupling.LastTsurfObs) < abs(TsurfAve - coupling.LastTsurfObs):
                coupling.start_coupling_again = True

            coupling.SwRadCof = 1.0
            coupling.LWRadCof = 1.0
            coupling.SW_correction = 0.0
            coupling.LW_correction = 0.0
            coupling.RadCoeff = 1.0
            coupling.coupling_failed = True

            # If surface temperature observation is missing, return to start of coupling and use radcof=1
        elif coupling.LastTsurfObs < -100:
            coupling.SwRadCof = 1.0
            coupling.LWRadCof = 1.0
            coupling.SW_correction = 0.0
            coupling.LW_correction = 0.0
            coupling.RadCoeff = 1.0
            coupling.coupling_failed = True
            coupling.start_coupling_again = True

        # If abnormal temperature value, return to start of coupling and use radcof=1
        elif TsurfAve < 170.0 or TsurfAve > 400.0 or coupling.coupling_failed:
            coupling.SwRadCof = 1.0
            coupling.LWRadCof = 1.0
            coupling.SW_correction = 0.0
            coupling.LW_correction = 0.0
            coupling.coupling_failed = True
            coupling.start_coupling_again = True
            coupling.RadCoeff = 1.0

        # If forecasted Tsurf greater than observed
        elif TsurfAve - coupling.LastTsurfObs > 0.1:

            # Save the guess the first time
            if coupling.TsurfNearestAbove < -100:
                coupling.TsurfNearestAbove = TsurfAve
                coupling.RadCoefNearestAbove = coupling.RadCoeff

            # Compare to the previous guess, if it is nearer observed temperature, save it
            elif coupling.TsurfNearestAbove - coupling.LastTsurfObs > TsurfAve - coupling.LastTsurfObs:
                coupling.TsurfNearestAbove = TsurfAve
                coupling.RadCoefNearestAbove = coupling.RadCoeff

            coupling.start_coupling_again = True

            # Calculate new guess, if there is one guess where observed temperature
            # is greater than forecasted and one where observed is smaller
            if coupling.TsurfNearestAbove > -100 and coupling.TsurfNearestBelow > -100:
                TDifAbove = coupling.TsurfNearestAbove - coupling.LastTsurfObs
                TDifBelow = coupling.LastTsurfObs - coupling.TsurfNearestBelow
                coupling.RadCoeff = coupling.RadCoefNearestAbove - TDifAbove / (TDifAbove + TDifBelow) * \
                                   (coupling.RadCoefNearestAbove - coupling.RadCoefNearestBelow)
            else:
                # Decrease radcof by half
                coupling.RadCoeff = 0.5 * coupling.RadCoeff
          

            # If not change, reset
            if abs(coupling.RadCoeff - coupling.RadCoeffPrevious) < 0.00005:
                coupling.TsurfNearestAbove = -9999
                coupling.TsurfNearestBelow = -9999
    
            if coupling.RadCoeff < 0.01:
                print("Coupling coefficient too small, coupling failed")
                coupling.RadCoeff = 1.0
                coupling.coupling_failed = True
                coupling.SwRadCof = 1.0
                coupling.LWRadCof = 1.0
                coupling.SW_correction = 0.0
                coupling.LW_correction = 0.0
    
            coupling.RadCoeffPrevious = coupling.RadCoeff
    
            # If forecasted Tsurf is smaller than observed
        elif coupling.LastTsurfObs - TsurfAve > 0.1:
    
            # Save guess the first time
            if coupling.TsurfNearestBelow < -100:
                coupling.TsurfNearestBelow = TsurfAve
                coupling.RadCoefNearestBelow = coupling.RadCoeff
    
            # Compare to the previous guess, if it is nearer observed temperature, save it
            elif coupling.TsurfNearestBelow - coupling.LastTsurfObs < TsurfAve - coupling.LastTsurfObs:
                coupling.TsurfNearestBelow = TsurfAve
                coupling.RadCoefNearestBelow = coupling.RadCoeff
    
            coupling.start_coupling_again = True
    
            # Calculate new guess, if there is one guess where observed temperature
            # is greater than forecasted and one where observed is smaller
            if coupling.TsurfNearestAbove > -100 and coupling.TsurfNearestBelow > -100:
                TDifAbove = coupling.TsurfNearestAbove - coupling.LastTsurfObs
                TDifBelow = coupling.LastTsurfObs - coupling.TsurfNearestBelow
                coupling.RadCoeff = coupling.RadCoefNearestAbove - TDifAbove / (TDifAbove + TDifBelow) * \
                                   (coupling.RadCoefNearestAbove - coupling.RadCoefNearestBelow)
            else:
                # Increase radcof
                coupling.RadCoeff = 2.0 * coupling.RadCoeff
    
            # If not change, reset
            if abs(coupling.RadCoeff - coupling.RadCoeffPrevious) < 0.00005:
                coupling.TsurfNearestAbove = -9999
                coupling.TsurfNearestBelow = -9999
    
            coupling.RadCoeffPrevious = coupling.RadCoeff
    
        else:
            if coupling.RadCoeff > 3.0:
                print("Coupling coefficient too big, coupling failed")
                coupling.coupling_failed = True
                coupling.RadCoeff = 1.0
                coupling.SwRadCof = 1.0
                coupling.LWRadCof = 1.0
                coupling.SW_correction = 0.0
                coupling.LW_correction = 0.0
    
            # Coupling was successful
            coupling.SW_correction = coupling.SwRadCof - 1.0
            coupling.LW_correction = coupling.LWRadCof - 1.0
            coupling.coupling_failed = False
            coupling.Coupling_iterations = -1
            coupling.TsurfNearestAbove = -9999.0
            coupling.TsurfNearestBelow = -9999.0
            coupling.RadCoeff = 1.0
            coupling.RadCoefNearestAbove = -9999.0
            coupling.RadCoefNearestBelow = -9999.0
            coupling.RadCoeffPrevious = 1.0


# Return to Celsius
    TsurfAve -= 273.16
    coupling.LastTsurfObs -= 273.16

