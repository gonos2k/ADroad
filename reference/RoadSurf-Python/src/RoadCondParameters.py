# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Class for variables related on calculating storage terms (snow, water, ice deposit)
"""
class RoadCondParameters:
    def __init__(self,DTSecs):
        self.WDampLim = 0.1  # Dry/Damp limit for water
        self.WWetLim = 0.9  # Damp/Wet limit for water
        self.WWearLim = 0.1  # Wear limit for water (below this only evaporation)
        self.Snow2IceFac = 0.5  # Snow to ice transition factor
        self.SnowIceRat = 0.0  # Snow-Ice ratio
        self.MissValI = -9999  # Missing integer value
        self.MissValR = -99.99  # Missing real value
        self.MinPrecmm = 0.05 * DTSecs / 3600.0  # MIN precipitation : mm/hour
        self.MinWatmms = 0.01 * DTSecs / 3600.0  # MIN water storage : mm
        self.MinSnowmms = 0.1 * DTSecs / 3600.0  # MIN snow storage : mm (accounts for snow "drift")
        self.MinDepmms = 0.01 * DTSecs / 3600.0  # MIN deposit storage : mm
        self.MinIcemms = 0.05 * DTSecs / 3600.0  # MIN ice storage : mm (MinPrec => condens won't freeze(?))
        self.MaxSnowmms = 100.0  # MAX snow storage : mm (plowed away if above)
        self.MaxDepmms = 2.0  # MAX deposit storage : mm
        self.MaxIcemms = 50.0  # MAX ice storage : mm
        self.MaxExtmms = 1.0  # MAX extra (surface) water content (mm)
        self.MaxWatmms = 2.0  # MAX water content
        self.AlbDry = 0.1  # Dry asphalt albedo
        self.AlbSnow = 0.6  # Snow albedo
        self.WatDens = 999.87  # Density of water at 0 C
        self.SnowDens = 100.0  # Density of snow (fresh, Oke p.44)
        self.IceDens = 920.0  # Density of ice (Oke p.44)
        self.DepDens = 920.0  # Density of deposit
        self.WatMHeat =333000.0  # Heat of ablation for water (J/kg)
        self.PorEvaF =  1.0  # Pore resistance factor for evaporation
        self.DampWearF = 0.5  # Damp surface (poer) wear factor
        self.TLimFreeze = -0.25  # Freezing limit (C)
        self.TLimMeltSnow = 0.25  # Melting limit for snow (C)
        self.TLimMeltIce = 0.25  # Melting limit for ice (C)
        self.TLimMeltDep = 1.25  # Melting limit for deposit (C)
        self.TLimDew = 0.25  # Dew/deposit formation limit (C)
        self.TLimColdH = -19.0  # Higher limit for cold ground temperature
        self.TLimColdL = -21.0  # Lower limit for cold ground temperature
        self.WetSnowFormR = 0.1  # Water to snow ratio for wet snow formation
        self.WetSnowMeltR = 0.6  # Water to snow ratio for wet snow melting
        self.PLimSnow = 0.3  # Precipitation interpretation : snow limit
        self.PLimRain = 0.7  # Precipitation interpretation : rain limit
        self.WetSnowFrozen = False  # True if wet snow is frozen
        self.freezing_limit_normal = -0.25  # Freezing limit
        self.snow_melting_limit_normal = 0.25  # Snow melting limit
        self.ice_melting_limit_normal = 0.25  # ice melting limit
        self.frost_melting_limit_normal = 1.25  # frost melting limit
        self.frost_formation_limit_normal = 0.25  # frost formation limit
        self.T4Melt_normal = 0.25  # limit for melting normal
        self.forceIceMelting = False  # True if ice melting is forced
        self.forceSnowMelting = False  # True if snow melting is forced
        self.CanMeltingChangeTemperature = False  # True if surface temperature is changed in melting
