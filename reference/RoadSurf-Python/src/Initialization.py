# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Functions related to model initialization
"""
from RoadSurfVariables import*
import BalanceModel
import BoundaryLayer

def initialize_model(timeStep,use_coupling,outputstep,
                     csv_data,lat,lon,forecast_start_time,start_time,end_time,initLen):
    # Initialize input data arrays
    modelInput = InputArrays.InputArrays(csv_data,timeStep,start_time,end_time,forecast_start_time)
    
    # Initialize output data arrays
    modelOutput = OutputArrays.OutputArrays(len(modelInput.time))
    
    # Initialize model settings
    settings =  ModelSettings.ModelSettings(len(modelInput.time),initLen,timeStep,use_coupling,outputstep)
    
    # Initialize physical parameters
    phy = PhysicalParameters.PhysicalParameters()
     
    # Initialize atmospheric properties
    atm = AtmVariables.AtmVariables(modelInput,initLen,timeStep)
    
    # Initialize surface properties
    surf = SurfaceVariables.SurfaceVariables(modelInput,atm)
    
    # Initialize ground properties
    ground = GroundVariables.GroundVariables(settings.NLayers,phy,atm,surf)
    
    # Initialize local parameters
    localParam = LocalParameters.LocalParameters(lat,lon,settings,modelInput,forecast_start_time,initLen)
    
    # Initialize coupling variables
    coupling = CouplingVariables.CouplingVariables(localParam,settings)    
    
    # Initialize road condition parameters
    condParam = RoadCondParameters.RoadCondParameters(timeStep)
    
    #Calculates Ground layers heat capacity and heat conductance
    BalanceModel.CalcHCapHCond(settings.NLayers, timeStep, phy, ground, atm)
    BalanceModel.calcCapDZCondDZ(settings.NLayers, ground)
    #Calculates latent heat flux and boundary layer conductance
    BoundaryLayer.CalcBLCondAndLE(surf, settings.DTSecs, surf.SrfWatmms, phy, atm)

    return (modelInput, modelOutput, phy, ground, surf, atm, coupling, settings, condParam, localParam)