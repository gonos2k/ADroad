# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development
Class for station specific variables
"""
class LocalParameters:
    def __init__(self,lat,lon,settings,modelInput,forecast_start_time,initLen):
        self.tair_relax = -9999.9  # tair for relaxation (not in use)
        self.VZ_relax = -9999.9  # wind speed for relaxation
        self.RH_relax = -9999.9  # relative humidity for relaxation
        self.couplingIndexI = self.initCouplingIndexI(settings,modelInput,forecast_start_time)  # index in input data where coupling starts
        self.couplingTsurf = self.initCouplingTsurf(settings,modelInput)  # surface temperature value used in coupling
        self.lat = lat  # latitude
        self.lon = lon  # longitude
        self.sky_view = -9999.9  # Sky view factor
        self.InitLenI = initLen  # The length of initialization period, input timesteps

    def initCouplingIndexI(self,settings,modelInput,forecast_start_time):
        if settings.use_coupling:
            index=int((forecast_start_time - modelInput.time[0]).total_seconds() / settings.DTSecs)
        else:
            index=-99
        return index
    
    def initCouplingTsurf(self,settings,modelInput):
        if settings.use_coupling:
            return modelInput.TSurfObs[self.couplingIndexI]
        else:
            return -9999.9