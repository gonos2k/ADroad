# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development
Class for input parameters
"""
class InputParameters:
    def __init__(self):
        self.NightOn = 0.0  # Beginning hour of night traffic (UTC)
        self.NightOff = 0.0  # Ending hour of night traffic (UTC)
        self.CalmLimDay = 0.0  # Minimum wind speed day (m/s)
        self.CalmLimNgt = 0.0  # Minimum wind speed night (m/s)
        self.TrfFricNgt = 0.0  # Traffic induced friction (night) (W/m2)
        self.TrFfricDay = 0.0  # Traffic induced friction day (W/m2)
        self.Grav = 0.0  # Gravitational acceleration (m/s2)
        self.SB_Const = 0.0  # Stefan-Boltzmann constant (W/m2K4)
        self.VK_Const = 0.0  # Von Karman's constant
        self.LVap = 0.0  # Latent heat of water vaporisation (J/kg)
        self.LFus = 0.0  # Latent heat of fusion (constant, not calculated)
        self.WatDens = 0.0  # Density of water at 0 C
        self.SnowDens = 0.0  # Density of snow (fresh, Oke p.44)
        self.IceDens = 0.0  # Density of ice (Oke p.44)
        self.DepDens = 0.0  # Density of deposit
        self.WatMHeat = 0.0  # Heat of ablation for water (J/kg)
        self.PorEvaF = 0.0  # Pore resistance factor for evaporation
        self.ZRefW = 0.0  # Wind reference height (m)
        self.ZRefT = 0.0  # Wind reference height (m)
        self.ZeroDisp = 0.0  # Zero displacement height (m)
        self.ZMom = 0.0  # Roughness factor for momentum (m)
        self.ZHeat = 0.0  # Roughness factor for heat (m)
        self.Emiss = 0.0  # Emissivity constant of the surface
        self.Albedo = 0.0  # Dry ground albedo
        self.Albedo_surroundings = 0.0  # albedo of surroundings
        self.MaxPormms = 0.0  # maximum porosity
        self.TClimG = 0.0  # Climatological temperature at the bottom layer
        self.DampDpth = 0.0  # Damping depth
        self.Omega = 0.0  # constant to calculate bottom layer temperature
        self.AZ = 0.0  # constant to calculate bottom layer temperature
        self.DampWearF = 0.0  # Damp surface (poer) wear factor
        self.AlbDry = 0.0  # Asphalt albedo
        self.AlbSnow = 0.0  # Snow albedo
        self.vsh1 = 0.0  # Heat capacity of surface layers (dry)
        self.vsh2 = 0.0  # Heat capacity of deep ground layers (dry)
        self.Poro1 = 0.0  # Porosity for surface layers
        self.Poro2 = 0.0  # Porosoty for deep ground layers
        self.RhoB1 = 0.0  # Bulk density for surface layers
        self.RhoB2 = 0.0  # Bulk density for ground layers
        self.Silt1 = 0.0  # Clay fraction for surface layers
        self.Silt2 = 0.0  # Clay fraction for deep ground layers
        self.freezing_limit_normal = 0.0  # Freezing limit (C)
        self.snow_melting_limit_normal = 0.0  # Melting limit for snow (C)
        self.ice_melting_limit_normal = 0.0  # Melting limit for ice (C)
        self.frost_melting_limit_normal = 0.0  # Melting limit for deposit (C)
        self.frost_formation_limit_normal = 0.0  # Dew/deposit formation limit (C)
        self.T4Melt_normal = 0.0  # Normal melting limit (C) surface class interpretation
        self.TLimColdH = 0.0  # Higher limit for cold ground temperature
        self.TLimColdL = 0.0  # Lower limit for cold ground temperature
        self.WetSnowFormR = 0.0  # Water to snow ratio for wet snow formation
        self.WetSnowMeltR = 0.0  # Water to snow ratio for wet snow melting
        self.PLimSnow = 0.0  # Precipitation interpretation : snow limit
        self.PLimRain = 0.0  # Precipitation interpretation : rain limit
        self.MaxSnowmms = 0.0  # MAX snow storage : mm (plowed away if above)
        self.MaxDepmms = 0.0  # MAX deposit storage : mm
        self.MaxIcemms = 0.0  # MAX ice storage : mm
        self.MaxExtmms = 0.0  # MAX extra (surface) water content (mm)
        self.MissValI = 0.0  # Missing value, integer
        self.MissValR = 0.0  # Missing value, real
        self.Snow2IceFac = 0.0  # Snow to ice transition factor
        self.MinPrecmm = 0.0  # MIN precipitation mm/hour
        self.MinWatmms = 0.0  # MIN water storage : mm
        self.MinSnowmms = 0.0  # MIN snow storage : mm (accounts for snow "drift")
        self.MaxWatmms = 0.0 # Maximum water storage
        self.WDampLim = 0.0  # Dry/Damp limit for water
        self.WWetLim = 0.0  # Damp/Wet limit for water
        self.WWearLim = 0.0  # Wear limit for water (below this only evaporation)
        self.MinDepmms = 0.0  # MIN deposit storage : mm
        self.MinIcemms = 0.0  # MIN ice storage : mm (MinPrec => condens won't freeze(?))
