# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Class for model physical parameters
"""
import math
class PhysicalParameters:
    def __init__(self):
              
        self.VK_Const = 0.4  # Von Karman's constant
        self.SB_const = 5.67e-8  # Stefan-Boltzmann constant (W/m2K4)
        self.ZRefW = 10.0  # Wind reference height (m)
        self.ZRefT = 2.0  # Temperature reference height (m)
        self.ZeroDisp = 0.0  # Zero displacement height (m)
        self.ZMom = 0.4000  # Roughness factor for momentum (m)
        self.ZHeat = 0.0010  # Roughness factor for heat (m)
        self.Grav = 9.81  # Gravitational acceleration (m/s2)
        self.Emiss = 0.95  # Emissivity constant of the surface
        self.Poro1 = 0.9  # porosity of asphalt
        self.Poro2 = 0.4  # porosity of ground layers
        self.vsh1 = 1.94e+06  # volumetric heat capacity of dry asphalt
        self.vsh2 = 1.28e+06  # volumetric heat capacity of dry ground
        self.LVap = 2.452E6  # latent heat of water vaporization
        self.LFus = 0.334E6  # latent heat of sublimation
        self.TClimG = 6.4  # Climatological temperature at the bottom ground layer
        self.MaxPormms = 1.0  # maximum water in asphalt pores
        self.DampDpth = 2.7  # Damping depth
        self.Omega = 2.0 * math.pi / 365.0  # Frequency of year variation (2pi/365)
        self.AZ = 0.6  # Amplitude at z(m+1), used to calculate bottom layer temperature
        self.Silt1 = 0.99  # Clay fraction for surface layers
        self.Silt2 = 0.8  # Clay fraction for deep ground layers
        self.RhoB1 = 2.11  # Bulk density for surface layers
        self.RhoB2 = 1.6  # Bulk density for deep ground layers
        self.logMom = math.log((self.ZRefW + self.ZMom) / self.ZMom)  #Help variable for boundary layer 
        self.logHeat = math.log((self.ZRefW + self.ZHeat) / self.ZHeat)  #Help variable for boundary layer 
        self.logCond = math.log((self.ZRefW - self.ZeroDisp +self.ZHeat)/self.ZHeat)  #Help variable for boundary layer 
        self.logUstar = math.log((self.ZRefW - self.ZeroDisp + self.ZMom)/self.ZMom)  #Help variable for boundary layer 
        self.Afc1 = 0.65-0.78*self.RhoB1 + 0.60*self.RhoB1*self.RhoB1  # Help variable used when calculating heat conductivity
        self.Bfc1 = 1.06*self.RhoB1  # Help variable used when calculating heat conductivity
        self.Cfc1 = 1 + 2.6/math.sqrt(self.Silt1)  # Help variable used when calculating heat conductivity
        self.Dfc1 = 0.03+0.1*self.RhoB1*self.RhoB1  # Help variable used when calculating heat conductivity
        self.Efc1 = 4  # Help variable used when calculating heat conductivity
        self.Afc2 = 0.65-0.78*self.RhoB2 + 0.60*self.RhoB2*self.RhoB2  # Help variable used when calculating heat conductivity
        self.Bfc2 = 1.06*self.RhoB2  # Help variable used when calculating heat conductivity
        self.Cfc2 = 1 + 2.6/math.sqrt(self.Silt2)  # Help variable used when calculating heat conductivity
        self.Dfc2 = 0.03+0.1*self.RhoB2*self.RhoB2  # Help variable used when calculating heat conductivity
        self.Efc2 = 4  # Help variable used when calculating heat conductivity
