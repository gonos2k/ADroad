# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Variables related to coupling
"""
class CouplingVariables:
    def __init__(self,localParam,settings):
        self.TmpSave = None  # Saved layer temperatures at the beginning of the coupling period
        self.Coupling_iterations = 0  # number of coupling interations
        self.TsurfNearestAbove = -9999.0  # Surface temperature at the end of coupling that is nearest 
                                          #to the observed temperature but is still above it
        self.TsurfNearestBelow = -9999.0  # Surface temperature at the end of coupling that is nearest
                                        #to the observed temperature but is still below it
        self.RadCoeff = 1.0  # used when determining radiation coefficient
        self.Down = False  # forecasted temperature has been greater than observed
        self.start_coupling_again = False  # if true, start coupling period again
        self.inCouplingPhase = False  # if true, model is currently in coupling phase
        self.VeryColdSave = False  # Saved veryCold value at the start of the coupling period
        self.RadCoefNearestAbove = -9999.0  # coefficient by which TsurfNearestAbove was achieved
        self.RadCoefNearestBelow = -9999.0  # coefficient by which TsurfNearestBelow was achieved
        self.RadCoeffPrevious = 1.0  # Radcoeff2 at previous iteration
        self.SWRadCof = 1.0  # radiation coefficient to use for short wave radiation
        self.LWRadCof = 1.0  # radiation coefficient to use for long wave radiation
        self.SW_correction = 0.0  # final radiation coefficient for short wave radiation
        self.LW_correction = 0.0  # final radiation coefficient for long wave radiation
        self.Tsurf_end_coup1 = -9999.0  # surface temperature at the end of first coupling period
        self.TSurfAveSave =-9999.0  # Saved surface temperature at the start of the coupling period
        self.SrfWatmmsSave = -9999.0  # Saved water storage at the start of the coupling period
        self.SrfIcemmsSave = -9999.0  # Saved ice storage at the start of the coupling period
        self.SrfIce2mmsSave = -9999.0  # Saved secondary ice storage at the start of the coupling period
        self.SrfDepmmsSave = -9999.0  # Saved deposit storage at the start of the coupling period
        self.SrfSnowmmsSave = -9999.0  # Saved snow storage at the start of the coupling period
        self.AlbedoSave = -9999.0  # Saved albedo at the start of the coupling period
        self.LastTsurfObs = localParam.couplingTsurf  # last surface temperature observation
        self.saveDatai = -9999  # i at the start of the coupling period
        self.SWSave = -9999.0  # Short wave radiaation at the start of coupling
        self.SWDirSave =-9999.0  # Direct short wave radiation at the start of coupling
        self.LWSave =-9999.0  # Long wave radiation at the start of coupling
        self.obsI =  localParam.couplingIndexI  # i index corresponding car observation time
        self.couplingStartI = self.initCouplingStartI(settings)  # i index to start coupling
        self.couplingEndI = self.initCouplingEndI(settings)  # i index to end coupling
        self.coupling_failed = False  # if true, coupling has been failed, continue without coupling

    #Timestep to start coupling    
    def initCouplingStartI(self,settings):
       if settings.use_coupling and self.obsI>-1:
           if self.obsI<=settings.coupling_minutes*60/settings.DTSecs:
               return 1
           else:
               return self.obsI-int(settings.coupling_minutes*60/settings.DTSecs)
           
    #Coupling end time step
    def initCouplingEndI(self,settings):
        if settings.use_coupling and self.obsI>-1:
            return self.obsI
        
