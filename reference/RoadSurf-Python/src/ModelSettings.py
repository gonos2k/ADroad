# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development
Class for variables related to model settins
"""
class ModelSettings:
    def __init__(self,dataLen,initLen,timeStep,use_coupling,outputStep):
        self.InitLenI = int(initLen*3600.0/timeStep)  # The length of initialization period, input timesteps
        self.SimLen = dataLen  # Length of simulation
        self.use_coupling = use_coupling  # coupling is used
        self.use_relaxation = False  # true if relaxation is used
        self.NLayers = 15  # number of ground layers
        self.DTSecs = timeStep  # time step in seconds
        self.simulation_failed = False  # true if simulation has been failed
        self.Tph = timeStep/3600.0  # time steps per hour
        self.NightOn = 19.0  # Beginning hour of night traffic (UTC)
        self.NightOff = 4.0  # Ending hour of night traffic (UTC)
        self.CalmLimDay = 1.5  # Minimum wind speed (m/s)
        self.CalmLimNgt = 0.4  # Minimum wind speed (m/s)
        self.TrfFricNgt = 5.0  # Traffic induced friction (W/m2)
        self.TrfFricDay = 10.0  # Traffic induced friction (W/m2)
        self.coupling_minutes = 180.0  # Length of the coupling period in minutes
        self.couplingEffectReduction = 4.0*3600  # Parameter used to calculate radiation coefficient after coupling
        self.outputStep = outputStep  # Frequency of output in minutes
