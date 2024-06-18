from dataclasses import dataclass
from general import iNAN, fNAN
from parallelization import ProcessControl
from enum import IntEnum

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
    
class LocWeight(IntEnum):
    """ Weighting schemes for domain localization. 2nd part refers to regulation used in the weights."""
    unit = 0 
    exponential = 1
    polynomial5 = 2
    polynomial5_mean = 3 
    polynomial5_single = 4 

@dataclass 
class RLocalizer: 
    """
    Basic localizer for domain localization. 
    """
    pe : ProcessControl
    projection : DistType
    cutoff_radius : float = fNAN
    weighting : LocWeight = LocWeight.unit 
    
    