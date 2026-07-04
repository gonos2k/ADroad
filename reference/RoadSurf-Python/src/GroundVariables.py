# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development
Ground related variables
"""
import numpy as np
import math
import BalanceModel

class GroundVariables:
    def __init__(self,array_size,phy,atm,surf):
        self.Albedo = 0.1  # surface albedo
        self.Albedo_surroundings=0.15
        self.HStor = 0.0  # Describes stored heat to the surface from previous time step
        self.condDZ = np.full(array_size,-9999.9)  # variable for temperature profile calculation
        self.capDZ = np.full(array_size,-9999.9)  # variable for temperature profile calculation
        self.WCont = self.initWCont(array_size)  # water content in ground levels
        self.VSH = np.full(array_size,-9999.9)  # Volumetric heat capacity of ground layers (J/(m^3K))
        self.HS = np.full(array_size,-9999.9)  # Heat capacity in intensity units (W/m^2K)
        self.CC = self.initCC(array_size,phy)  # ground heat conductivity
        self.ZDpth = self.initDepth(array_size)  # Depths of ground layers
        self.Tmp = self.initTemp(atm, phy, array_size,surf)  # Temperatures for each layer
        self.TmpNw = self.Tmp.copy()  # Updated temperature profile
        self.DyC = self.initDyC(array_size)  # Layers thicknesses for heat capacity calculation
        self.DyK = self.initDyK(array_size)  # Layer thicknesses for heat conductivity calculation
        self.GCond = np.full(array_size+1,-9999.9)  # Heat conductivity divided by layer height
        self.GroundFlux = 0.0  # ground heat flux (between 2nd and 3rd layers)

    def initDepth(self,NLayers):
        ZAdd = 0.02
        ZDpth = np.full(NLayers+1,0.0)
        
        # Calculates layer depths so that the thicknesses increase with depth
        for I in range(NLayers):
            ZDpth[I + 1] = ZDpth[I] + 0.0103 * 1.4 ** (I+1 - 1) + ZAdd
        return ZDpth
    
    def initDyC(self,NLayers):
        DyC = np.zeros(NLayers + 1, dtype=float)
    
        DyC[0] = (self.ZDpth[1] - self.ZDpth[0]) / 2.0
        
        for j in range(1, NLayers):
            DyC[j] = (self.ZDpth[j + 1] - self.ZDpth[j - 1]) / 2.0
        return DyC
            
    def initDyK(self,NLayers):
        DyK = np.diff(self.ZDpth[:NLayers + 1])
        return DyK
    
    def initWCont(self,NLayers):
        wcont=np.full(NLayers,0.3)
        wcont[0]=0.01
        wcont[1]=0.01
        return wcont
    
    def initCC(self,NLayers,phy):
        CC=np.zeros(NLayers)
        for I in range(NLayers):
            if I <= 1:
                CC[I] = phy.Afc1 + phy.Bfc1 * self.WCont[I] - \
                    (phy.Afc1 - phy.Dfc1) * math.exp(-(phy.Cfc1 * self.WCont[I]) ** phy.Efc1)
            else:
                CC[I] = phy.Afc2 + phy.Bfc2 * self.WCont[I] - \
                    (phy.Afc2 - phy.Dfc2) * math.exp(-(phy.Cfc2 * self.WCont[I]) ** phy.Efc2)
        return CC
            
    def initTemp(self,atm, phy, NLayers,surf):
        Tmp=np.full(NLayers+2,-9999.9)
        Tmp[0] = atm.Tair  # 0th index is air temperature
    
        # First four are the same as observed surface temperature, if observation is available
        if surf.TsurfOBS > -100:
            Tmp[1:5] = surf.TsurfOBS
        else:
            Tmp[1:5] = atm.Tair
    
        juld = BalanceModel.calculate_julian_day()
        
        Tmp[-1] = phy.TClimG + phy.AZ * np.sin(phy.Omega * \
                        juld + phy.Omega * (-170) - (self.ZDpth[-1] / phy.DampDpth))
    
        
        # Temperature of the rest of the layers approaches linearly to the climatological value
        Tmp[5:] = Tmp[4] + (Tmp[-1] - Tmp[4]) \
            / (self.ZDpth[-1] - self.ZDpth[3]) * (self.ZDpth[4:] - self.ZDpth[3])
    
        return Tmp
    
