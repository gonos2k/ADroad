# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development
#Model output arrays, saved every time step
"""
import numpy as np
class OutputArrays:
    def __init__(self, array_size):
        self.TsurfOut = np.full(array_size,-9999.9)  # Road surface temperature (C)
        self.SnowOut = np.full(array_size,-9999.9)  # snow storage (water equivalent mm)
        self.WaterOut = np.full(array_size,-9999.9)  # water storage (mm)
        self.IceOut = np.full(array_size,-9999.9)  # ice storage (water equivalent mm)
        self.DepositOut = np.full(array_size,-9999.9)  # deposit storage (water equivalent mm)
        self.Ice2Out = np.full(array_size,-9999.9)  # secondary ice storage