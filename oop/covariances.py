import numpy as np 
from general import fNAN, iNAN, check_init
from enum import IntEnum
from abc import ABC, abstractmethod 
from dataclasses import dataclass
from localization import RLocalizer
from parallelization import ProcessControl

class ObsErrType(IntEnum):
    """
    This contains options for observational error distribution.
    """
    gauss = 0 
    laplace = 1

def contra2matrix(func):
    def wrapper(self,contra):
        b = np.reshape(contra, (-1,1)) if np.ndim(contra)==1 else contra
        Ab = func(self,b)
        return np.reshape(Ab,(-1,)) if np.ndim(contra)==1 else Ab 
    return wrapper 

def co2matrix(func):
    def wrapper(self,co):
        b = np.reshape(co, (1,-1)) if np.ndim(co)==1 else co 
        Ab = func(self,b)
        return np.reshape(co, (-1,)) if np.ndim(co)==1 else co 
    return wrapper 
        
            
class Covariance(ABC):
    """ Abstract class represting error distribution. 
    
    Methods 
    -------
    left_multiply : 
        Multiply on left with (inverse/sqrt) covariance matrix. 
    right_multiply : 
        Multiply on right with (inverse/sqrt) covariance matrix.
    
    Attributes
    ----------
    pe : parallelization.ProcessControl object
        comm_filter object with process containing this covariance. 
    time : float 
        Time for which covariance is valid.
    is_initialized : bool 
        Bool indicating whether time has been set. 
    variance : ndarray
        Array with diagonal of covariance matrix. 
        
    """
    
    @abstractmethod 
    def copy(self):
        """ Create hard copy of current object."""
        
    @abstractmethod
    def inverse(self):
        """ Return inverse of covariance. """
        
    @abstractmethod
    def sqrt(self):
        """ Return square-root of covariance."""
        
    @abstractmethod
    def init(self, time, **kwargs):
        """ 
        Create covariance from input data. 
        
        odata : dict 
            Dictionary with different observation information. 
        """
        
    @abstractmethod 
    def element(self, indices):
        """ 
        Return matrix elements
        
        Parameters
        ----------
        indices : iter of (2,)-tuplets
            Indices of elements sought. 
        """
        
    @abstractmethod 
    def left_multiply(self, input):
        """ Product covariance @ vector/matrix. """
       
    @abstractmethod  
    def right_multiply(self, input):
        """ Product vector/matrix @ covariance. """
    
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
    pe : ProcessControl
    localizer : RLocalizer 
    error_family : ObsErrType = ObsErrType.gauss
    is_initialized : bool = False
    
    def copy(self):
        cov = DiagonalObsCovariance(pe=self.pe, localizer=self.localizer, 
                                    error_family=self.error_family)
        if self.is_initialized:
            cov.init(self.time, variance=self.variance)
        return cov
            
    @check_init
    def inverse(self):
        cov = self.copy()
        cov.init(self.time, variance=1/self.variance)
        return cov
        
    @check_init 
    def sqrt(self):
        cov = self.copy()
        cov.init(self.time, variance=np.sqrt(self.variance))
        return cov
    
    def init(self, da_time, **kwargs):
        """ 
        Create covariance from input data. 
        
        odata : dict 
            Dictionary with different observation information. 
        """
        self.variance = kwargs['variance']
        self.time = da_time
        self.is_initialized = True
        
    @check_init
    @co2matrix    
    def right_multiply(self, input):
        A = np.reshape(self.variance,(1,-1))
        return input * A
    
    @check_init
    @contra2matrix
    def left_multiply(self, input):
        A = np.reshape(self.variance, (-1,1))
        return A * input
    
    @check_init
    def element(self, indices):
       ndim = np.ndim(indices)
       indices = np.array(indices, dtype=int).reshape((-1, np.size(indices,-1)))
       elements = np.array([self.variance[i] if i==j else 0. for i,j in indices], dtype=float)
       return elements[0] if ndim==1 else elements
        
    
    
    
    
    