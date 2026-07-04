# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

This is research version of FMI's road weather model RoadSurf. There are
some differences to the Fortran version. Parts of the code are converted
from fortran to Python with a large language model (LLM). The same model
physics still applies, see Fortarn model's physics documentation here:

Karsisto, V. E. 2024: RoadSurf 1.1: open-source road weather model library, 
Geosci. Model Dev., 17, 4837–4853, https://doi.org/10.5194/gmd-17-4837-2024
"""

from RoadSurfVariables import*
import readInputcsv
import BalanceModel
import Initialization
import InputOutput
import Coupling
import Storage
import ModRadiation
import Cond 
from matplotlib import pyplot as plot
from datetime import datetime,timedelta
import writecsv
import pandas as pd
import matplotlib.dates as mdates
#import Relaxation #Relaxation not fully implemented in this version

def roadModelOneStep(input_idxI, phy, ground, surf, atm,settings, coupling, 
                     modelInput, condParam,localParam):
   
    #Determine wheter the precipitation is rain or snow and add to storage
    Storage.PrecipitationToStorage(settings,condParam,modelInput.PrecPhase[input_idxI],atm,surf)

    #Make radiation corrections based on sky view and local horizon angles
    if localParam.sky_view<1.0 and localParam.sky_view>-0.01:
      ModRadiation.ModRadiationBySurroundings(modelInput,ground,localParam,input_idxI)
     
    #Calculate temperature profile one time step forward
    #Checks also for melting (can affect temperature)
    BalanceModel.BalanceModelOneStep(modelInput.SW[input_idxI], modelInput.LW[input_idxI], 
                            phy, ground, surf, atm, settings, coupling, modelInput,
                            input_idxI,condParam)
   
    wearF=WearingFactors.WearingFactors()
    # ************* WEAR FACTORS
    Cond.WearFactors(condParam, settings.Tph, surf, wearF)
    
    #Calculate storage terms 
    Cond.RoadCond(phy.MaxPormms, surf, atm, settings,condParam,wearF)
    
    #Calculate albedo
    ground.Albedo=Cond.CalcAlbedo(surf, condParam)

#Define model settings
init_length=48 #Initialization period length in hours
forecast_length=60 #Forecast period length in hours
forecast_start=datetime(2021,3,4,3,0,0) #Start of the forecast phase
start_time=forecast_start-timedelta(hours=init_length) #Simulation start
end_time=forecast_start+timedelta(hours=forecast_length) #Simulation end
timeStep=30.0 #Model time step in seconds
use_coupling=False #PATCHED: no-coupling fixture
outputStep=15 #written output frequency in minutes

#Coordinates of the road point
lat=62.246
lon=25.769

# Input data path
file_path = '../example_data/test_input.csv'

# Read in data
csv_data = readInputcsv.read_csv_data(file_path)
csv_data["timestamp"]=[t.timestamp() for t in csv_data["time"]]

#Initialization
(modelInput, modelOutput, phy, ground, surf, atm, coupling, settings, condParam, localParam) = \
    Initialization.initialize_model(timeStep,use_coupling,outputStep,
                     csv_data,lat,lon,forecast_start,start_time,end_time,init_length)

i=0

#Simulation starts
while i<settings.SimLen-1 and settings.simulation_failed == False:
    
    #Check input for bad values
    InputOutput.CheckValues(modelInput, i, settings, surf, localParam)
    
    #Check if coupling is on
    if settings.use_coupling:
        #Check if in coupling phase, save variables if at the start of the
        # coupling phase,
        #Load saved variables if coupling is started again.
        #After coupling, calculate radiation coefficients
        i=Coupling.CouplingOperations1(i, coupling, surf, settings, ground, 
                                       modelInput, condParam, localParam)
    #set current values 
    InputOutput.SetCurrentValues(i, modelInput, atm, settings, surf, coupling,ground)
  
    #If relaxation is used
    #if settings.use_relaxation:
         #Smooth t2m, rh and wind values when moving from initialization phase
         #to forecasting phase
     #    Relaxation.relaxation_operations(i, atm, settings,ground)
    #Calculate temperature profile and storage values one timestep forward
    roadModelOneStep(i, phy, ground, surf, atm,settings, coupling, modelInput, 
                     condParam, localParam)
   

    #Save output
    InputOutput.SaveOutput(modelOutput, i, surf)
    Coupling.CheckEndCoupling(i, settings, coupling, surf) 
    
    i=i+1
   
  
# If simulation is not failed, make calculations for the last value
if not settings.simulation_failed:
    # Make still calculation for i=SimLen (the last value)
    
    # Set last input values as interpolated values
    InputOutput.lastValues(modelInput, atm, settings, ground, surf)
    
    # Calculate temperature profile and storage values one time step forward
    roadModelOneStep(i, phy, ground, surf, atm,
                     settings, coupling, modelInput, condParam,
                     localParam)
  
    InputOutput.SaveOutput(modelOutput, i, surf)

#Write output to csv
writecsv.write_to_csv(modelOutput, modelInput, "../output/testi_output.csv", outputStep)

# --------SIMULATION END----------------------------
#Plotting

fig, ax = plot.subplots(2, 1, figsize=(10,10),sharex=True)

# Plot model output storages
ax[0].plot(modelInput.time, modelOutput.SnowOut, "r", label="snow")
ax[0].plot(modelInput.time, modelOutput.IceOut, "cyan", label="ice")
ax[0].plot(modelInput.time, modelOutput.WaterOut, "b", label="water")
ax[0].plot(modelInput.time, modelOutput.DepositOut, "k", label="deposit")
ax[0].set_ylabel("Storage (water equivalent mm)")
ax[0].legend()
ax[0].axvline(x=forecast_start,linestyle='--',color='k',alpha=0.5)


myFmt = mdates.DateFormatter('%d-%b-%y')
ax[0].xaxis.set_major_formatter(myFmt)

#Mask too small values for plotting with nans
mask=modelInput.TSurfObs>-100
# Plot model output temperature
ax[1].plot(modelInput.time, modelOutput.TsurfOut, "r", label="road temperature")
ax[1].plot(modelInput.time[mask], modelInput.TSurfObs[mask], "--b", label="observed road temperature")
ax[1].set_ylabel("Temperature (°C)")
ax[1].legend()
ax[1].axvline(x=forecast_start,linestyle='--',color='k',alpha=0.5)
ax[1].axhline(y=0,linestyle='--',color='grey')
ax[1].xaxis.set_major_formatter(myFmt)


plot.show()
