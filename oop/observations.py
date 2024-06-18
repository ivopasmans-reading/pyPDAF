"""This file is part of pyPDAF

Copyright (C) 2022 University of Reading and
National Centre for Earth Observation

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import pyPDAF.PDAF as PDAF
from abc import ABC, abstractmethod
from general import fNAN, iNAN, check_init

class Observation(ABC):
    """
    Abstract class representing a specific type of observation.
    
    Methods 
    -------
    create_windows :
        Create observation windows. 
    create_windows_from_model :
        Create observation windows matching model time steps.
    obs_op : 
        Convert state vector by observation operator.
    obs_op_lin :
        Linearized version of observe.
    obs_op_adj : 
        Adjoint of tl_observer.
    init : 
        Initialize observation operator for a specific time. 
    deallocate : 
        Deallocate memory after completing DA step. 

    Attributes
    ----------
    pe : parallelization.ProcessControl object
        comm_filter object with process carrying out the observation. 
    obs_covariance : covariances.ObsCovariance object 
        Object representing observational error covariance including localization. 
    doassim : bool 
        Flag indicating whether this observation type should be assimilated. 
    is_initialized : bool 
        Flag indicating that observation operator is initialized for specific time
        and obs_op can be used. 
        
    """
    
    @abstractmethod
    def obs_op(self, time, state_p, ostate=None):
        """
        Convert state vector by observation operator.

        Parameters
        ----------
        time : float
            Current time.
        state_p : ndarray
            Part state vector on this process. 
        ostate : ndarray
            
            
        Returns
        -------
        ostate
        
        """   
        
    def obs_op_lin(self, time, state_p, ostate=None):
        """ 
        Linearized version of observe.
        
        Parameters
        ----------
        time : float
            Current time.
        state_p : ndarray
            Part state vector on this process. 
        ostate : ndarray
            
            
        Returns
        -------
        ostate
        
        """
        if self.doassim:
            msg = f"Observation type {self.obs_id} has no tangent linear operator."
            self.pe.throw_error(msg)
        
    def obs_op_adj(self, time, state_p, ostate=None):
        """ 
        Adjoint of tl_observer.
        
        Parameters
        ----------
        time : float
            Current time.
        state_p : ndarray
            Part state vector on this process. 
        ostate : ndarray
            
            
        Returns
        -------
        ostate
        
        """
        if self.doassim:
            msg = f"Observation type {self.obs_id} has no adjoint operator."
            self.pe.throw_error(msg) 
    
    def create_windows(self,windows={}):
        """
        Create observation windows. 

        windows : dict 
            Dictionary with assimilation time window (as float)
            as keys and window start and end w.r.t. to this 
            time as value. 
            
        """
        self.windows = {}
        odata = self.read_obs()
        #Filter out empty windows.
        for wtime,window in windows.items():
            in_window = np.logical_and(odata['time']>wtime+np.min(window),
                                       odata['time']<=wtime+np.max(window))
            if np.any(in_window):
                self.windows = {**self.windows, wtime:window}
        
    def create_windows_from_model(self, time_init, final_step, dt):
        """
        Create observation windows matching model time steps.

        time_init : float 
            Model initialization time. 
        dt : float 
            Model time step. 
            
        """
        wtimes = np.arange(1, final_step+1) * dt + time_init
        windows = [(wtime,(-dt,0)) for wtime in wtimes]
        self.create_windows(dict(windows))
        
    def next_obs_time(self, time_now):
        """ 
        Find first DA window after current time.
        """
        times = np.array([float(key) for key in self.windows])
        times = times[times > time_now]
        return np.min(times) if len(times)>0 else None
        
    @abstractmethod 
    def init(self, id, time):
        """
        Retrieve observations and create observation operator for a
        specific time. 

        id : int > 0
            Observation type index. 
        time : float 
            DA time. 
            
        """
            
    def _check_odata(self, odata):
        """ Check whether all required fields are present in observational data. """
        #Collect missing fields. 
        keys = set(['observed','coord','time','variance'])
        keys = keys - set(odata.keys())
         
        #Create error message. 
        msg = ""
        for key in keys:
            msg += f"Key {key} missing in the observational data for observation {self.obs_id}.\n"
            
        if len(keys)>0:
            self.pe.throw_error(msg)
        
    def deallocate(self):
        """ Deallocate observations. """
        if self.is_initialized:
            PDAF.omi_deallocate_obs(self.obs_id)
            self.dim_obs = 0
            self.is_initialized = False
        
class PointObservation(Observation):
    """ 
    Class representing observations representing values at a single point. 
    
    Methods
    -------
    read_obs : function 
        Function that reads observations for this time. 
    interpolator : function 
        Given geographic coordinates, retrieve indices and weights in state_p. 
    obs_covariance : covariances.ObsCovariance object 
        Object representing observational error covariance including localization. 
    
    """
    
    def __init__(self, pe,  obs_covariance, obs_reader, interpolator):
        """
        Constructor 
        
        Parameters 
        ----------
        pe : parallelization.ProcessControl object
            comm_filter object contains process used. 
        obs_reader : function 
            Function that reads observations for this time into dictionary. 
        interpolator : function 
            Given geographic coordinates, retrieve indices and weights in state_p. 
        obs_covariance : covariances.ObsCoveriance object 
            Object representing observational error covariance including localization. 
        time : float 
            Time for which observation operator was intialized. 
        is_initialized : bool 
            Bool indicating that observation operator was initialized. 
            
        """
        self.doassim = True
        self.pe = pe 
        self.read_obs = obs_reader
        self.interpolator = interpolator
        self.covariance = obs_covariance 
        self.is_initialized = False
    
    @check_init
    def obs_op(self, state_p, ostate=None):
        
        if self.doassim and ostate is None:
            ostate = np.zeros((self.dim_obs,), dtype=float, order='F')
        if self.doassim:
            ostate = PDAF.omi_obs_op_gridpoint(self.obs_id, state_p, ostate)
        return ostate
        
    def init(self, id, time):
        #Numerical index of this observation operator. Must be >=1.
        self.obs_id = id
        self.dim_obs = 0 
        self.time = time 
        #Select observation in this observation window. 
        odata = {'observed':np.array([],dtype=float), 'coord':np.array([],dtype=float),
                 'stencil_index':np.array([],dtype=int),'stencil_weight':np.array([],dtype=float),
                }
        matches = [wtime for wtime in self.windows.keys() if np.isclose(wtime,time)]
        if len(matches)==0:
            n_coord, n_stencil = 1,1
        elif len(matches)==1:
            window = matches[0] + np.array(self.windows[matches[0]])
            #Read observations. 
            odata = self.read_obs(window)
            self._check_odata(odata)
            
            #Determine indices in state_p and weights of observation functional.
            odata['stencil_index'], odata['stencil_weight'] = self.interpolator(odata['coord'])
            n_coord, n_stencil = np.size(odata['coord'],1), np.size(odata['stencil_index'],1)
            in_p = np.any(odata['stencil_weight']!=0.0, axis=1)
            for key in odata:
                odata[key] = odata[key][in_p]
        else:
            self.pe.throw_error("For each time at most 1 observation window may be defined.")
            
        #Save values in this object. 
        for key in ['observed','coord','stencil_index','stencil_weight']:
            setattr(self, key, odata[key])
        self.covariance.init(self.time, **odata)
        
        #Pass on to Fortran
        if len(odata['observed'])==0:
            odata = {'observed':np.ones((1,))*fNAN, 'coord':np.ones((1,n_coord)) * fNAN,
                     'stencil_index':np.zeros((1,n_stencil)),'stencil_weight':np.zeros((1,n_stencil)),
                    }
            inv_variance = np.zeros((1,)) 
        else:
            inv_variance = 1./self.covariance.variance
            
        #Pass everything on to the Fortran code. 
        PDAF.omi_set_doassim(self.obs_id, int(self.doassim and len(matches)>0))
        PDAF.omi_set_disttype(self.obs_id, int(self.covariance.localizer.projection))
        PDAF.omi_set_ncoord(self.obs_id, n_coord)
        PDAF.omi_set_id_obs_p(self.obs_id, np.asfortranarray(odata['stencil_index'].T, dtype=int)+1)
        PDAF.omi_set_icoeff_p(self.obs_id, np.asfortranarray(odata['stencil_weight'].T, dtype=float))
        PDAF.omi_set_obs_err_type(self.obs_id, int(self.covariance.error_family)) 
        PDAF.omi_set_use_global_obs(self.obs_id, int(self.covariance.localizer.cutoff_radius < 0.))
        #TODO: PDAF.omi_set_domainsize(self.obs_id, self.domainsize)
        
        self.dim_obs = PDAF.omi_gather_obs(self.obs_id, np.asfortranarray(odata['observed'],dtype=float), 
                                           np.asfortranarray(inv_variance, dtype=float), 
                                           np.asfortranarray(odata['coord'].T,dtype=float), 
                                           self.covariance.localizer.cutoff_radius)
        
        self.is_initialized = True
      
        return self.dim_obs

class ObsReader(ABC):
    """ Abstract class to deal with reading observational data. """
    
    @abstractmethod 
    def read(self, window):
        """ Return observations in time window. """
        
    @abstractmethod 
    def times(self):
        """ Retrieve all times for which observations are available. """

class DictObsReader:
    """
    Class to read observations from given dictionary. 

    Parameters 
    ----------
    data : dict 
        Dictionary with data that must include keys time, coord, value, variance.

    """
    
    def __init__(self, pe, data):
        """ Constructor. """
        self.pe = pe
        self.data = data
        
    def __call__(self, window=None):
        """ Read observations in window. """
        #Times 
        times = self.data['time']
        if window is None:
            return {'time':times}

        in_window = np.logical_and(times>min(window), times<=max(window))
        return dict([(key,value[in_window]) for key, value in self.data.items()])
    
    