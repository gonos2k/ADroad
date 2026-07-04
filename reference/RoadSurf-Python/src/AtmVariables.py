# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Atmoshperic variables used in the model
Many are updated each timestep
"""

class AtmVariables:
    def __init__(self,inputData,initLen,DTSecs):
        self.Tair = inputData.Tair[0]  # air temperature (C)
        self.VZ = inputData.VZ[0]    # wind speed (m/s)
        self.Tdew = inputData.Tdew[0] # dew point temperature (C)
        self.Rhz = inputData.Rhz[0]  # relative humidity (%)
        self.PrecInTStep = inputData.prec[0]/3600.0*DTSecs  # precipitation in model time step
        self.TairInitEnd = inputData.Tair[int(initLen*3600/DTSecs)] # Air temperature at the end of initialization (C)
        self.VZInitEnd = inputData.VZ[int(initLen*3600/DTSecs)]    # wind speed at the end of initialization (m/s)
        self.RhzInitEnd = inputData.Rhz[int(initLen*3600/DTSecs)]  # relative humidity at the end of initialization (%)
        self.BLCond = -9999.9       # boundary layer conductance
        self.RNet = -9999.9         # Net radiation (W/m2)
        self.LE_Flux = -9999.9      # latent heat flux (W/m2)
        self.RainIntensity = -9999.9  # rain intensity (mm/h)
        self.SnowIntensity = -9999.9  # snow intensity (mm/h)
        self.TairR = -9999.9      # Air temperature from forecast at the end of initialization (C)
        self.VZR = -9999.9        # wind speed from forecast at the end of initialization (m/s)
        self.RhzR = -9999.9       # relative humidity from forecast at the end of initialization (%)
        self.CalmLim = 0.4    # Minimum wind speed
        self.SensibleHeatFlux = -9999.9  # sensible heat flux (W/m2)
        self.RainmmTS = -9999.9   # amount of rain in time step
        self.SnowmmTS = -9999.9   # amount of snow in time step
        self.SnowType = -99     # snow type (wet or dry)
        self.PrecType = -99     # Precipitation type (1=Water, 2=Sleet, 3=Snow)
