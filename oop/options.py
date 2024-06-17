from enum import IntEnum

#Nan value for integers. 
iNAN = -999
#Nan value for floats. 
fNAN = -9.99e9


class FilterType(IntEnum):
    """
    Filter options for PDAF. 
    """
    seek = 0
    seik = 1 
    enkf = 2
    lseik = 3
    etkf = 4 
    letkf = 5
    estkf = 6 
    lestkf = 7 
    local_enkf = 8 
    netf = 9 
    lnetf = 10 
    lknetf = 11 
    pf = 12 
    genobs = 100 
    var3d = 200 

class ScreenOutput(IntEnum):
    """
    This class contains options to print PDAF 
    output to screen.  
    """
    quiet = 0 
    standard = 1
    timing = 2
    
class PdafError(IntEnum):
    """ 
    This class contains possible ways PDAF can fail.
    """
    none = 0
    
class DistType(IntEnum):
    """
    This class contains the geographic projection used. 
    """
    cartesian = 0 
    periodic_cartesian = 1
    simplified_geographic = 2
    haversine_geographic = 3
    cartesian_factorized = 10 
    periodic_cartesian_factorized = 11 
    simplified_geographic_factorized = 12 
    haversine_geographic_factorized = 13

class ObsErrType(IntEnum):
    """
    This contains options for observational error distribution.
    """
    gauss = 0 
    laplace = 1
    
class LocWeight(IntEnum):
    """ Weighting schemes for domain localization. 2nd part refers to regulation used in the weights."""
    unit = 0 
    exponential = 1
    polynomial5 = 2
    polynomial5_mean = 3 
    polynomial5_single = 4 
