# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Check input for abnormal values, set current values to atmvarialbes,
save output values
"""
from RoadSurfVariables import*

#Check input for abnormal values
def CheckValues(modelInput, i, settings, surf, localParam):
    if (modelInput.Tair[i] < -90.0 or modelInput.Tair[i] > 100.0 or
        modelInput.Tdew[i] < -90 or modelInput.Tdew[i] > 100.0 or
        modelInput.Rhz[i] < -0.1 or modelInput.Rhz[i] > 120.0 or
        modelInput.VZ[i] < -1.0 or modelInput.VZ[i] > 100.0 or
        modelInput.SW[i] < -0.1 or modelInput.SW[i] > 4000.0 or
        modelInput.LW[i] < -0.1 or modelInput.LW[i] > 1000.0 or
        modelInput.prec[i] < -0.1 or modelInput.prec[i] > 500.0):

        print("BAD input value! ", modelInput.Tair[i], modelInput.Tdew[i], modelInput.RHz[i], 
              modelInput.VZ[i], modelInput.SW[i], modelInput.LW[i], modelInput.prec[i])
        settings.simulation_failed = True

    if (localParam.sky_view < 1.0 and localParam.sky_view > -0.01):
        if (modelInput.SW_dir[i] < -0.1 or modelInput.SW_dir[i] > 4000.0 or
            modelInput.LW_net[i] < -1000.0 or modelInput.LW_net[i] > 1000.0):

            print("BAD input value: SW_dir,LW_net", modelInput.SW_dir[i], modelInput.LW_net[i])
            settings.simulation_failed = True

    if (modelInput.SW_dir[i] > modelInput.SW[i]):
        modelInput.SW_dir[i] = modelInput.SW[i]

    if (surf.TsurfAve < -100.0 or surf.TsurfAve > 100.0):
        print("Abnormal surface temperature", surf.TsurfAve, i, localParam.lat, localParam.lon)
        settings.simulation_failed = True

#set values to Tair etc
def SetCurrentValues(i, modelInput, atm, settings, surf,coupling,ground):
    atm.Tair = modelInput.Tair[i]
    atm.Tdew = modelInput.Tdew[i]
    atm.VZ = modelInput.VZ[i]
    atm.Rhz = modelInput.Rhz[i]
    atm.PrecInTStep = modelInput.prec[i] / 3600 * settings.DTSecs
    ground.Tmp[0] = atm.Tair
    
    if i <= settings.InitLenI:
      
        if modelInput.TSurfObs[i] > -100.0:
            
            if not settings.use_coupling or (i < coupling.couplingStartI):
               
                surf.TSurfObs = modelInput.TSurfObs[i]
                ground.Tmp[1] = surf.TSurfObs
                ground.Tmp[2] = surf.TSurfObs
                
                surf.TsurfAve = (ground.Tmp[1] + ground.Tmp[2]) / 2.0
            else:
                surf.TSurfObs = -9999.0
        else:
            surf.TSurfObs = -9999.0

#Save to output arrays
def SaveOutput(modelOutput, i, surf):
    # Arrays for model input data
   
    modelOutput.SnowOut[i] = surf.SrfSnowmms
    modelOutput.WaterOut[i] = surf.SrfWatmms
    modelOutput.IceOut[i] = surf.SrfIcemms
    modelOutput.Ice2Out[i] = surf.SrfIce2mms
    modelOutput.DepositOut[i] = surf.SrfDepmms
    modelOutput.TsurfOut[i] = surf.TsurfAve
    
# Sets last input values for interpolated values
def lastValues(modelInput, atm, settings, ground, surf):
    # Variables for model settings
    # modelInput, atm, ground, surf should be instances of respective classes or structures

    # Get values from modelInput for the last time step (settings%SimLen)
    atm.Tair = modelInput.Tair[settings.SimLen - 1]
    atm.Tdew = modelInput.Tdew[settings.SimLen - 1]
    atm.VZ = modelInput.VZ[settings.SimLen - 1]
    atm.Rhz = modelInput.Rhz[settings.SimLen - 1]
    atm.PrecInTStep = modelInput.prec[settings.SimLen - 1] / 3600 * settings.DTSecs

    surf.TsurfAve = (ground.Tmp[1] + ground.Tmp[2]) / 2.0  # Average temperature of the first two layers
