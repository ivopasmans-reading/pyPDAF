import numpy as np 
from options import fNAN, iNAN, ObsErrType
from abc import ABC, abstractmethod 
from dataclasses import dataclass
from localization import RLocalizer
 
class Covariance(ABC):
    pass
    
@dataclass 
class DiagonalObsCovariance(Covariance):
    """ 
    Class representing diagonal observation error covariance. 
    
    Attributes
    ----------
    localizer : localization.ObsLocalizer object 
        Object containing information for R-localization. 
    error_family : options.ObsErrType 
        Probability distribution family used for observational errors. 
    """
    localizer : RLocalizer 
    error_family : ObsErrType = ObsErrType.gauss
    
    