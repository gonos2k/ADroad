# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Relaxation is not fully implented in this version
"""
import math

def relaxation_operations(i, atm, settings, ground):
    init_li = settings.InitLenI*3600.0/settings.DTSecs
    DTs = settings.DTSecs

    if i > init_li:
        print("here 1", i, atm.Tair, atm.TairInitEnd)
        atm.Tair = atm.Tair - (atm.TairR - atm.TairInitEnd) * \
                   (2.71828 ** (-(DTs * i - DTs * init_li) / (4.0 * 3600.0)))
        ground.Tmp[0] = atm.Tair

        atm.VZ = atm.VZ - (atm.VZR - atm.VZInitEnd) * \
                 (2.71828 ** (-(DTs * i - DTs * init_li) / (4.0 * 3600.0)))

        atm.Rhz = atm.Rhz - (atm.RhzR - atm.RhzInitEnd) * \
                   (2.71828 ** (-(DTs * i - DTs * init_li) / (4.0 * 3600.0)))

        if atm.Rhz > 100.0:
            atm.Rhz = 100.0
        print("here 2", i, atm.Tair, atm.TairInitEnd)
    atm.Tdew=calc_tdew(atm)

def calc_tdew(atm):
    # Constants
    AFact = 0.61078  # e in kPa
    Alphai = 21.875  # over ice
    Betai = 265.5   # over ice
    Alphaw = 17.269  # over water
    Betaw = 237.3   # over water

    # Variables
    if atm.Tair >= 0:
        Alpha = Alphaw
        Beta = Betaw
    else:
        Alpha = Alphai
        Beta = Betai

    EPrSat = AFact * math.exp(Alpha *atm.Tair / (atm.Tair + Beta))
    Epr = 0.01 * atm.Rhz * EPrSat
    XX = math.log(Epr / AFact)
    Tdew = Beta * XX / (Alpha - XX)

    return Tdew