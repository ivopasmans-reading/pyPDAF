from dataclasses import dataclass
from options import iNAN, fNAN, LocWeight, DistType

@dataclass 
class RLocalizer: 
    """
    Basic localizer for domain localization. 
    """
    projection : DistType
    cutoff_radius : float = fNAN
    weighting : LocWeight = LocWeight.unit 
    