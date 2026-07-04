# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Functions reltaed to modifying incoming radiation based on sky view factor
and local horizon angles
"""
import SunPosition

def ModRadiationBySurroundings(modelInput, ground, localParam, i):

    # Calculation is based on paper: "Parametrization of orographic effects
    # on surface radiation in HIRLAM, Senkova et al. 2007"

    # Diffuse radiation is the difference between global and direct sw radiation
    dif_SW = modelInput.SW[i] - modelInput.SW_dir[i]

    # Calculate upward lw radiation from net lw radiation and downwelling lw radiation
    # This is the radiation emitted by the road surroundings
    LW_surroundings = modelInput.LW_net[i] - modelInput.LW[i]

    # Calculate sun position
    sun_elevation, sun_azim,jde=SunPosition.SunPosition(modelInput, localParam, i)
   
    # If sun is above horizon
    if sun_elevation > 0.0:

        # Take local horizon angle that corresponds sun azimuth
        #horizon_in_sun_dir = 0.0
        azim_idx = round(sun_azim)
        if azim_idx == 360:
            azim_idx = 0
        horizon_in_sun_dir = modelInput.local_horizons[azim_idx]

        # Location is in shadow if sun elevation is lower than local horizon angle
        if horizon_in_sun_dir > sun_elevation:
            shadow_fac = 0.0
        else:
            shadow_fac = 1.0
        # Reduces direct solar radiation if the sun is behind an obstacle
        modelInput.SW_dir[i] = modelInput.SW_dir[i] * shadow_fac
        # Reflected radiation from surroundings
        SW_ref = ground.Albedo_surroundings * modelInput.SW_dir[i] + \
                 ground.Albedo_surroundings * dif_SW
        # Total incoming diffuse short wave radiation
        dif_SW = localParam.sky_view * dif_SW + (1.0 - localParam.sky_view) * SW_ref

        # Total incoming short wave radiation
        modelInput.SW[i] = dif_SW + modelInput.SW_dir[i]

    # Total incoming long wave radiation
    modelInput.LW[i] = localParam.sky_view * modelInput.LW[i] + \
                       (1.0 - localParam.sky_view) * (-LW_surroundings)

