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
from options import fNAN, iNAN

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
        self.windows = windows
        
    def create_windows_from_model(self, time_init, total_steps, dt):
        """
        Create observation windows matching model time steps.

        time_init : float 
            Model initialization time. 
        dt : float 
            Model time step. 
            
        """
        wtimes = np.arange(0, total_steps) * dt + time_init
        windows = [(wtime,(0,dt)) for wtime in wtimes]
        self.create_windows(dict(windows))
        
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
        
    def _dummy_odata(self, ncoords=1, nstencil=1):
        """ 
        Dummy initialize in case no observations are present in window. 
        """
        y = np.zeros(1, order='F')
        ocoords = np.ones((ncoords,1), dtype=float, order='F') * fNAN
        stencil_index = np.zeros((nstencil,1), dtype=int, order='F') 
        stencil_weight = np.zeros((nstencil,1), dtype=float, order='F') 
        inverse_var = np.zeros(1, order='F', dtype=float) 

        return y, ocoords, inverse_var, stencil_index, stencil_weight
        
    def deallocate(self):
        """ Deallocate observations. """
        PDAF.omi_deallocate_obs(self.obs_id)
        self.is_initialized = False
        
class PointObservation(Observation):
    """ 
    Class representing observations representing values at a single point. 
    
    Methods
    -------
    obs_reader : function 
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
            
        """
        self.doassim = True
        self.pe = pe 
        self.read_obs = obs_reader
        self.interpolator = interpolator
        self.covariance = obs_covariance 
    
    def obs_op(self, state_p, ostate=None):
        #if not np.isclose(time, self.last_initialized):
        #    self.pe.throw_error(f"Observation operator for time {time} not yet initialized.")
        if self.doassim and ostate is None:
            ostate = np.zeros((self.dim_obs,), dtype=float, order='F')
        if self.doassim:
            print('IP',np.shape(state_p))
            return PDAF.omi_obs_op_gridpoint(self.obs_id, state_p, ostate)
        
    def init(self, id, time):
        #Numerical index of this observation operator. Must be >=1.
        self.obs_id = id
        #Select observation in this observation window. 
        matches = [wtime for wtime in self.windows.keys() if np.isclose(wtime,time)]
        if len(matches)==1:
            window = matches[0] + np.array(self.windows[matches[0]])
            #Read observations. 
            odata = self.read_obs(window)
            stencil_index, stencil_weight = self.interpolator(odata['coord'])
            self.stencil_shape = (np.size(odata['coord'],1), np.size(stencil_weight,1)) #C-order
            #Count how many points on this comm_filter.
            in_p = np.sum(stencil_weight, axis=1) > 0
            for key in odata:
                odata[key] = odata[key][in_p]
            self.stencil_index = stencil_index[in_p]
            self.stencil_weight = stencil_weight[in_p]
            self.ocoords = odata['coord']
        else:
            self.pe.throw_error("For each time at most 1 observation window may be defined.")
            
        #Find interpolation stencil and determine which observations are on this comm_filter
        if len(matches)==0: 
            y, ocoords, inverse_var, stencil_index, stencil_weight = self._dummy_odata()
        elif sum(in_p)==0:
            y, ocoords, inverse_var, stencil_index, stencil_weight = self._dummy_odata(*self.stencil_shape)
        else:
            y = np.asfortranarray(odata['value'], dtype=float)
            ocoords = np.asfortranarray(odata['coord'].T, dtype=float)
            stencil_index = np.asfortranarray(stencil_index.T, dtype=int) + 1
            stencil_weight = np.asfortranarray(stencil_weight.T, dtype=float)
            inverse_var = np.asfortranarray(1./odata['variance'], dtype=float)
            
        #Pass everything on to the Fortran code. 
        PDAF.omi_set_doassim(self.obs_id, int(self.doassim and len(matches)>0))
        PDAF.omi_set_disttype(self.obs_id, int(self.covariance.localizer.projection))
        PDAF.omi_set_ncoord(self.obs_id, np.size(ocoords, 0))
        PDAF.omi_set_id_obs_p(self.obs_id, stencil_index)
        PDAF.omi_set_icoeff_p(self.obs_id, stencil_weight)
        PDAF.omi_set_obs_err_type(self.obs_id, int(self.covariance.error_family)) 
        PDAF.omi_set_use_global_obs(self.obs_id, int(self.covariance.localizer.cutoff_radius < 0.))
        #TODO: PDAF.omi_set_domainsize(self.obs_id, self.domainsize)
        
        print('IP',np.shape(y),np.shape(inverse_var),np.shape(ocoords),
              np.shape(stencil_index),np.shape(stencil_weight))
        self.dim_obs = PDAF.omi_gather_obs(self.obs_id, y, inverse_var, ocoords, 
                                           self.covariance.localizer.cutoff_radius)
      
        return self.dim_obs

class DictObsReader:
    """
    Class to read observations from given dictionary. 

    Parameters 
    ----------
    data : dict 
        Dictionary with data that must include keys time, coord, value, variance.

    """
    
    def __init__(self, data):
        """ Constructor. """
        self.data = data
        
    def __call__(self, window):
        """ Return observations in time window. """
        times = self.data['time']
        in_window = np.logical_and(times>=min(window), times<max(window))
        return dict([(key,value[in_window]) for key, value in self.data.items()])
    
class DepreciatedOBS:
    """observation information and user-supplied routines

    Attributes
    ----------
    delt_obs : int
        time step interval for observations
    dim_obs : int
        dimension size of the observation vector
    dim_obs_p : int
        dimension size of the PE-local observation vector
    disttype : int
        type of distance computation to use for localization
    doassim : int
        whether to assimilate this observation type
    domainsize : ndarray
        size of domain for periodicity (<=0 for no periodicity)
    i_obs : int
        index of the observation type
    icoeff_p : ndarray
        2d array for interpolation coefficients for obs. operator
    id_obs_p : ndarray
        indices of process-local observed field in state vector
    ivar_obs_p : ndarray
        vector of process-local inverse observation error variance
    n_obs : int
        number of observation types
    ncoord : int
        number of coordinate dimension
    nrows : int
        number of rows in ocoord_p
    obs_err_type : int
        type of observation error
    obs_p : ndarray
        vector of process-local observations
    ocoord_p : ndarray
        2d array of process-local observation coordinates
    rms_obs : float
        observation error standard deviation (for constant errors)
    use_global_obs : int
       Whether to use (1) global full obs. or
       (0) obs. restricted to those relevant for a process domain
    """

    n_obs = 0

    def __init__(self, typename, mype_filter,
                 nx, doassim, delt_obs, rms_obs):
        """constructor

        Parameters
        ----------
        typename : string
            name of the observation type
        mype_filter : int
            rank of the PE in filter communicator
        nx : ndarray
            grid size of the model domain
        doassim : int
            whether to assimilate this observation type
        delt_obs : int
            time step interval for observations
        rms_obs : float
            observation error standard deviation (for constant errors)
        """
        OBS.n_obs += 1

        self.i_obs = OBS.n_obs

        assert OBS.n_obs >= 1, 'observation count must start from 1'

        if (mype_filter == 0):
            print(('Assimilate observations:', typename))

        self.doassim = doassim
        self.delt_obs = delt_obs
        self.rms_obs = rms_obs

        # Specify type of distance computation
        # 0=Cartesian 1=Cartesian periodic
        self.disttype = 0

        # Number of coordinates used for distance computation
        # The distance compution starts from the first row
        self.ncoord = len(nx)

        # Allocate process-local index array
        # This array has as many rows as required
        # for the observation operator
        # 1 if observations are at grid points;
        # >1 if interpolation is required
        self.nrows = 1

        # Size of domain for periodicity for disttype=1
        # (<0 for no periodicity)
        if self.i_obs == 1:
            self.domainsize = np.zeros(self.ncoord)
            self.domainsize[0] = nx[1]
            self.domainsize[1] = nx[0]
        else:
            self.domainsize = None

        # Type of observation error: (0) Gauss, (1) Laplace
        self.obs_err_type = None

        # Whether to use (1) global full obs.
        # (0) obs. restricted to those relevant for a process domain
        self.use_global_obs = 1

        self.icoeff_p = None

    def init_dim_obs(self, step, dim_obs, local_range,
                     mype_filter, nx, nx_p):
        """intialise PDAFomi and getting dimension of observation vector

        Parameters
        ----------
        step : int
            current time step
        dim_obs : int
            dimension size of the observation vector
        local_range : float
            range for local observation domain
        mype_filter : int
            rank of the PE in filter communicator
        nx : ndarray
            integer array for grid size
        nx_p : ndarray
            integer array for PE-local grid size
        """
        obs_field = self.get_obs_field(step, nx)

        # Count valid observations that
        # lie within the process sub-domain
        pe_start = nx_p[-1]*mype_filter
        pe_end = nx_p[-1]*(mype_filter+1)
        obs_field_p = obs_field[:, pe_start:pe_end]
        assert tuple(nx_p) == obs_field_p.shape, \
               'observation decomposition should be the same as' \
               ' the model decomposition'
        cnt_p = np.count_nonzero(obs_field_p > -999.0)
        self.dim_obs_p = cnt_p

        # Initialize vector of observations on the process sub-domain
        # Initialize coordinate array of observations
        # on the process sub-domain
        if self.dim_obs_p > 0:
            self.set_obs_p(nx_p, obs_field_p)
            self.set_id_obs_p(nx_p, obs_field_p)
            self.set_ocoord_p(obs_field_p, pe_start)
            self.set_ivar_obs_p()
        else:
            self.obs_p = np.zeros(1)
            self.ivar_obs_p = np.zeros(1)
            self.ocoord_p = np.zeros((self.ncoord, 1))
            self.id_obs_p = np.zeros((self.nrows, 1))

        self.set_PDAFomi(local_range)

    def set_obs_p(self, nx_p, obs_field_p):
        """set up PE-local observation vector

        Parameters
        ----------
        nx_p : ndarray
            PE-local model domain
        obs_field_p : ndarray
            PE-local observation field
        """
        obs_field_tmp = obs_field_p.reshape(np.prod(nx_p), order='F')
        self.obs_p = np.zeros(self.dim_obs_p)
        self.obs_p[:self.dim_obs_p] = obs_field_tmp[obs_field_tmp > -999]

    def set_id_obs_p(self, nx_p, obs_field_p):
        """set id_obs_p

        Parameters
        ----------
        nx_p : ndarray
            PE-local model domain
        obs_field_p : ndarray
            PE-local observation field
        """
        self.id_obs_p = np.zeros((self.nrows, self.dim_obs_p))
        obs_field_tmp = obs_field_p.reshape(np.prod(nx_p), order='F')
        cnt0_p = np.where(obs_field_tmp > -999)[0] + 1
        assert len(cnt0_p) == self.dim_obs_p, 'dim_obs_p should equal cnt0_p'
        self.id_obs_p[0, :self.dim_obs_p] = cnt0_p

    def set_ocoord_p(self, obs_field_p, offset):
        """set ocoord_p

        Parameters
        ----------
        obs_field_p : ndarray
            PE-local observation field
        offset : int
            PE-local offset starting from rank 0
        """
        self.ocoord_p = np.zeros((self.ncoord, self.dim_obs_p))
        ix, iy = np.where(obs_field_p.T > -999)
        self.ocoord_p[0, :self.dim_obs_p] = ix + 1 + offset
        self.ocoord_p[1, :self.dim_obs_p] = iy + 1

    def set_ivar_obs_p(self):
        """set ivar_obs_p
        """
        self.ivar_obs_p = np.ones(
                                self.dim_obs_p
                                )/(self.rms_obs*self.rms_obs)

    def get_obs_field(self, step, nx):
        """retrieve observation field

        Parameters
        ----------
        step : int
            current time step
        nx : ndarray
            grid size of the model domain

        Returns
        -------
        obs_field : ndarray
            observation field
        """
        obs_field = np.zeros(nx)
        if self.i_obs == 1:
            obs_field = np.loadtxt(f'inputs_online/obs_step{step}.txt')
        else:
            obs_field = np.loadtxt(f'inputs_online/obsB_step{step}.txt')
        return obs_field

    def set_PDAFomi(self, local_range):
        """set PDAFomi obs_f object

        Parameters
        ----------
        local_range : double
            lcalization radius (the maximum radius used in this process domain)
        """
        print (self.obs_p)
        PDAF.omi_set_doassim(self.i_obs, self.doassim)
        PDAF.omi_set_disttype(self.i_obs, self.disttype)
        PDAF.omi_set_ncoord(self.i_obs, self.ncoord)
        PDAF.omi_set_id_obs_p(self.i_obs, self.id_obs_p)
        if self.domainsize is not None:
            PDAF.omi_set_domainsize(self.i_obs, self.domainsize)
        if self.obs_err_type is not None:
            PDAF.omi_set_obs_err_type(self.i_obs, self.obs_err_type)
        if self.use_global_obs is not None:
            PDAF.omi_set_use_global_obs(self.i_obs, self.use_global_obs)
        if self.icoeff_p is not None:
            PDAF.omi_set_icoeff_p(self.i_obs, self.icoeff_p)

        self.dim_obs = PDAF.omi_gather_obs(self.i_obs,
                                          self.obs_p,
                                          self.ivar_obs_p,
                                          self.ocoord_p,
                                          local_range)

    def obs_op(self, step, state_p, ostate):
        """convert state vector by observation operator

        Parameters
        ----------
        step : int
            current time step
        state_p : ndarray
            PE-local state vector
        ostate : ndarray
            state vector transformed by identity matrix
        """
        if (self.doassim == 1):
            ostate = PDAF.omi_obs_op_gridpoint(self.i_obs, state_p, ostate)
        return ostate

    def init_dim_obs_l(self, localization, domain_p, step, dim_obs, dim_obs_l):
        """intialise local observation vector

        Parameters
        ----------
        localization : TYPE
            Description
        domain_p : int
            index of current local analysis domain
        step : int
            current time step
        dim_obs : int
            dimension of observation vector
        dim_obs_l : int
            dimension of local observation vector

        Returns
        -------
        dim_obs_l : int
            dimension of local observations
        """
        return PDAF.omi_init_dim_obs_l(self.i_obs, localization.coords_l,
                                      localization.loc_weight,
                                      localization.local_range,
                                      localization.srange)

    def localize_covar(self, localization, HP_p, HPH, coords_p):
        """localze covariance matrix

        Parameters
        ----------
        localization : `Localization.Localization`
            the localization object
        HP_p : ndarray
            matrix HPH
        HPH : ndarray
            PE local part of matrix HP
        coords_p : ndarray
            coordinates of state vector elements
        """
        PDAF.omi_localize_covar(self.i_obs, localization.loc_weight,
                               localization.local_range,
                               localization.srange,
                               coords_p, HP_p, HPH)

    def deallocate_obs(self):
        """deallocate PDAFomi object

        Parameters
        ----------
        step : int
            current time step
        """
        PDAF.omi_deallocate_obs(self.i_obs)
