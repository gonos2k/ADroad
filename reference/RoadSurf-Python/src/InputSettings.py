# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development
Class for simulation settings
"""
class InputSettings:
    def __init__(self):
        self.SimLen = 0  # Length of simulation
        self.use_coupling = 0  # 1 if coupling is used
        self.use_relaxation = 0  # 1 if relaxation is used
        self.force_tsurf = 0  # 1 if surface temperature is forced to given value full simulation
        self.DTSecs = 0.0  # time step in seconds
        self.NLayers = 0  # Number of ground layers
        self.coupling_minutes = 0  # coupling length in minutes
        self.couplingEffectReduction = 0.0  # parameter used to calculate radiation coefficient after coupling
        self.outputStep = 0  # model output frequency in minutes
