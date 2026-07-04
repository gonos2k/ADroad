# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Class for storage wearing variables
"""
class WearingFactors:
    def __init__(self):
        self.SnowTran = 0.0  # Snow transform factor : mm/timestep
        self.SnowTran2 = 0.0  # Snow transform factor night
        self.SnowTranDef = 0.0  # default value for transform
        self.DepWear = 0.0  # Deposit wear : mm/timestep
        self.IceWear = 0.0  # Ice wear : mm/timestep
        self.IceWear2 = 0.0  # Ice wear2 : mm/timestep
        self.IceWearNight = 0.0  # Ice wear night : mm/timestep
        self.IceWear2Night = 0.0  # Ice wear2 night : mm/timestep
        self.IceWearSW = 0.0  # Ice wear by sun : mm/timestep
        self.IceWear2SW = 0.0  # Ice wear2 by sun : mm/timestep
        self.WatWear = 0.0  # Water wear : mm/timestep
