# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development
Class for timestep interpolated model input data
"""
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plot
class InputArrays:
    def __init__(self, data,DTsecs,start_time,end_time,forecast_start_time):
        self.time = self.createTimeArray(DTsecs,start_time,end_time)
        self.timestamp=np.array([t.timestamp() for t in self.time])
        self.Tair = np.interp(self.timestamp, data["timestamp"], np.array(data["air_temperature"]))  # air temperature (C)
        self.Tdew = np.interp(self.timestamp, data["timestamp"], np.array(data["dewpoint"]))  # Dew point temperature (C)
        self.VZ = np.interp(self.timestamp, data["timestamp"], np.array(data["windSpeed"]))  # wind speed (m/s)
        self.Rhz = self.CalcRH()  # relative humidity (%)
        self.prec = np.interp(self.timestamp, data["timestamp"], np.array(data["prec"]))  # precipitation (mm/h)
        self.SW = np.interp(self.timestamp, data["timestamp"], np.array(data["SW"]))  # incoming short wave radiation (W/m2)
        self.LW = np.interp(self.timestamp, data["timestamp"], np.array(data["LW"]))  # incoming long wave radiation (W/m2)
        self.SW_dir = np.full(len(self.time),200.0)  # direct short wave radiation (W/m2)
        self.LW_net = np.full(len(self.time),300.0)  # net long wave radiation (W/m2)
        self.TSurfObs = self.initTsurfObs(data,forecast_start_time)  # surface temperature (C)
        self.PrecPhase = np.full(len(self.time),-9999)  # precipitation phase (Hail=6; FreezingRain=5; FreezingDrizzle=4; Snow=3; Sleet=2; Rain=1; Drizzle=0)
        self.local_horizons = np.full(360,-9999.9)  # local horizon angles
        
    def createTimeArray(self,DTsecs,start_time,end_time):
        
        # Calculate the number of 30-second intervals between start and end
        total_seconds = (end_time - start_time).total_seconds()
        num_intervals = int(total_seconds / DTsecs)
        
        # Create a time array with values for every 30 seconds
        time_array = [start_time + timedelta(seconds=i * DTsecs) for i in range(num_intervals)]
        
        # Convert the list of datetime objects to a NumPy array if needed
        time_array = np.array(time_array)
        return time_array

#Calculate relative humidity
    def CalcRH(self):
        Alphaw = 17.269  # over water
        Alphai = 21.875  # over ice
        Betaw = 237.3    # over water
        Betai = 265.5    # over ice
        AFact = 0.61078  # e in kPa
    
        Alpha = np.zeros_like(self.Tair)
        Beta = np.zeros_like(self.Tair)
    
        condition = self.Tair >= 0.0
        Alpha[condition] = Alphaw
        Alpha[~condition] = Alphai
        Beta[condition] = Betaw
        Beta[~condition] = Betai
    
        EsatT = AFact * np.exp(Alpha * self.Tair / (self.Tair + Beta))
    
        EsatTD = AFact * np.exp(Alpha * self.Tdew / (self.Tdew+ Beta))
        result = np.minimum((EsatTD / EsatT) * 100.0, 100.0)        

        return result
    
    #Surface temperature observations, fileld with -9999 in the forecast phase
    def initTsurfObs(self,data,forecast_start_time):

        tsurf = np.full(len(self.time), -9999.9)
        valid_mask = self.time <= forecast_start_time

        tsurf[valid_mask] = np.interp(self.timestamp[valid_mask], data["timestamp"], np.array(data["troad"]))
        return tsurf

        




