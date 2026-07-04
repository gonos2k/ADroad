# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Class for road surface variables
"""
class SurfaceVariables:
    def __init__(self,inputData,atm):
        self.SrfWatmms = 0.0  # water storage (mm)
        self.SrfSnowmms = 0.0  # snow storage (water equivalent mm)
        self.SrfIcemms = 0.0  # ice storage (water equivalent mm)
        self.SrfIce2mms = 0.0  # Secondary ice storage (water equivalent mm)
        self.SrfDepmms = 0.0  # deposit storage (water equivalent mm)
        self.Q2Melt = 0.0  # Amount of heat needed for ice/snow to melt (W/m2) / actual amount of heat (W/m2) used to melt ice/snow
        self.T4Melt = 0.0  # Limit temperature for ice/snow to melt
        self.TrfFric = 5.0  # Surface heating caused by traffic
        self.EvapmmTS = 0.0  # Evaporation
        self.VeryCold = False  # true if ground temperature is cold
        self.WearSurf = True  # true if storage terms can be reduced by traffic wearing
        self.TsurfOBS = inputData.TSurfObs[0]  # surface temperature observation
        self.TsurfAve = self.initTsurfAve(atm)  # average temperature of the first two layers
        
    def initTsurfAve(self,atm):
        if self.TsurfOBS>-99.0:
            return self.TsurfOBS
        else:
            return atm.Tair