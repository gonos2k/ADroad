# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Functions for sun position calculation
"""
from math import atan, atan2, sin, cos, asin, acos
def SunPosition(modelInput, localParam, i):
    # Calculate solar elevation and azimuth angle
    JDE = JulianEphemerisDay(modelInput, i)
    elevation_angle, azimuth_angle = calcElevationAzimuth(JDE, localParam.lat, localParam.lon)

    return elevation_angle, azimuth_angle,JDE

def calcElevationAzimuth(JDE, lat, lon):
    """"
    Method
!     ------
!
!     Jean Meeus, Astronomical Algorithms
!     Chapters:
!       - 11, Sidereal Time at Greenwich
!       - 12, Transformation of Coordinates
!       - 21, Nutation and the Obliquity of the Ecliptic
!       - 24, Solar Coordinates
!
!     cos(solar zenith angle) = sin(latitude)*sin(solar declinationination)
!             + cos(latitude)*cos(solar declinationination)*cos(hour angle)
!     solar elevation angle = 90 - solar zenith angle
!
!     Azimuth angle calculated as in 
!     https://www.pveducation.org/pvcdrom/properties-of-sunlight/azimuth-angle
!
!     cos(solar azimuth angle)=(sin(solar declinationination)*cos(latitude)-
!                               cos(solar declinationination)*sin(latitude)*cos(hour_angle))
!                               /cos(sun elevation)
!
!     If elevation angle <=0, elevation angle and azimuth angle are set to
!     -9999.9
!
!
!-----------------------------------------------------------------------
"""

   
    # Constants
    pi = 4 * atan(1.0)
    Dyr = 365.25

    # T, Julian centuries since J2000.0
    T = (JDE - 2451545.0) / (Dyr * 100.)

    # ml, geometric mean longitude of the Sun
    ml = 280.46645 + 36000.76983 * T + 0.0003032 * T**2

    if ml < 0.:
        ml = ml - 360. * (int(ml / 360.) - 1.)
    if ml > 360.:
        ml = ml - 360. * int(ml / 360.)

# ----ma, mean anomaly of the Sun
    ma = 357.52910 + 35999.05030 * T - 0.0001559 * T**2 - 0.00000048 * T**3

    if ma < 0.:
        ma = ma - 360. * (int(ma / 360.) - 1.)
    if ma > 360.:
        ma = ma - 360. * int(ma / 360.)
 
# ----ecc, eccentricity of the earth's orbit
    ecc = 0.016708617 - 0.000042037 * T - 0.0000001236 * T**2

# ----sunc, Sun's equation of center
    sunc = (1.913600 - 0.004817 * T - 0.000014 * T**2) * sin(ma * pi / 180.) \
         + (0.019993 - 0.000101 * T) * sin(2. * ma * pi / 180.) \
         + 0.000290 * sin(3. * ma * pi / 180.)

# ----al, apparent longitude of the Sun
    al = ml + sunc - 0.00569 - 0.00478 * sin((125.04 - 1934.136 * T) * pi / 180.)
    al = al * pi / 180.
   
# ----tilt, mean obliquity of the ecliptic
    tilt = 23.43929111 - 0.013004166 * T - 0.001638888 * T**2 \
         + 0.005036111 * T**3
    eps = tilt + 0.00256 * cos((125.04 - 1934.136 * T) * pi / 180.)
    eps = eps * pi / 180.

# ----ra, apparent right ascension of the Sun
    ra = atan2(cos(eps) * sin(al), cos(al))

    if ra < 0.:
        ra = ra - 2. * pi * (int(ra / (2. * pi)) - 1.)
    if ra > 2. * pi:
        ra = ra - 2. * pi * int(ra / (2. * pi))

# ----declination, apparent declination of the Sun
    declination = asin(sin(eps) * sin(al))
   
# ----stG, mean sidereal time at Greenwich
    stG = 280.46061837 + 360.98564736629 * (JDE - 2451545.0) \
        + 0.000387933 * T**2 - T**3 / 38710000.

    if stG < 0.:
        stG = stG - 360. * (int(stG / 360.) - 1.)
    if stG > 360.:
        stG = stG - 360. * int(stG / 360.)

    stG = stG * pi / 180.
   
# ----trigonometry
    cos_declination = cos(declination)
    sin_declination = sin(declination)
    lat_radians = pi * lat / 180.
    sin_lat = sin(lat_radians)
    cos_lat = cos(lat_radians)
    cos_dec_lat = cos_declination * cos_lat
    sin_dec_lat = sin_declination * sin_lat
 
# ----hour_angle_corr, local hour angle
    hour_angle_corr = (stG + lon * pi / 180. - ra)
    if ra < 0.:
        hour_angle_corr = hour_angle_corr - 2. * pi * (int(hour_angle_corr / (2. * pi)) - 1.)
    if ra > 2. * pi:
        hour_angle_corr = hour_angle_corr - 2. * pi * int(hour_angle_corr / (2. * pi))
   
# ---- solar zenith angle
    cosah = cos(hour_angle_corr)
    cos_elev = sin_dec_lat + cos_dec_lat * cosah
  
    if 1.0 <= cos_elev < 1.001:
        cos_elev = 1.0
        chi = 0.0
    elif 1.001 <= cos_elev:
        print('Problem with zenith angle at lat/lon:', lat, lon)
        raise ValueError('Zenith angle issue')
    elif -1.001 <= cos_elev <= -1.0:
        cos_elev = -1.0
        chi = pi
    else:
        chi = acos(cos_elev)
   
    elevation_angle = 90.0 - chi * (180. / pi)

# ----hour angle correction
    if hour_angle_corr < 0.:
        hour_angle_corr = 2 * pi + hour_angle_corr
    elif hour_angle_corr > 2 * pi:
        hour_angle_corr = hour_angle_corr - 2 * pi
# ----solar azimuth angle
    # Calculate only if elevation angle > 0
   
    if elevation_angle > 0:
        cosele = cos((pi / 2.0) - chi)
        precos = 0.0

        if -0.0001 <= cosele < 0.0001:
            azimuth_angle = -9999.9
        else:
            precos = (sin_declination * cos_lat - cos_declination * sin_lat * cosah) / cosele

            if 1.0 <= precos < 1.001:
                precos = 1.0
                azimuth_angle = 0.0
            elif precos >= 1.001:
                print('Problem with azimuth angle at lat/lon:', lat, lon)
                raise ValueError('Azimuth angle issue')
            elif -1.001 <= precos <= -1.0:
                precos = -1.0
                azimuth_angle = pi
            else:
                azimuth_angle = acos(precos)

            if hour_angle_corr < pi:
                azimuth_angle = 2 * pi - azimuth_angle

            azimuth_angle = azimuth_angle * (180. / pi)
    else:
        azimuth_angle = -9999.9
    #    elevation_angle = -9999.9
 
    return elevation_angle, azimuth_angle

def JulianEphemerisDay(modelInput, idx):

    """
   Method
!     ------
!
!     Jean Meeus, Astronomical Algorithms
!     Chapters:
!       - 7, Julian Day

    """
    # Constants
    Dyr = 365.25
    
    # Extract date and time components from modelInput
    mmyr = modelInput.time[idx].year
    mmmon = modelInput.time[idx].month
    mmday = modelInput.time[idx].day
    mmhr = modelInput.time[idx].hour
    mmmin = modelInput.time[idx].minute
    mmsec = modelInput.time[idx].second
    
    # Calculate Julian Ephemeris Day
    if mmmon <= 2:
        yr = mmyr - 1
        mo = mmmon + 12
    else:
        yr = mmyr
        mo = mmmon

    day = mmday + mmhr / 24. + mmmin / (24. * 60.) + mmsec / (24. * 60. * 60.)

    A = int(yr / 100.)
    B = 2. - A + int(A / 4.)

    JDE = int(Dyr * (yr + 4716)) + int(30.6001 * (mo + 1.)) + day + B - 1.5245e3
   
    return JDE
