from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from general import fNAN, iNAN, check_init
from enum import IntEnum
import pyPDAF.PDAF as PDAF
import numpy as np
from parallelization import ProcessControl
from covariances import Covariance
from inflation import Inflator

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

def identity_state_pdaf(dim_p, state_p):
    """
    Dummy for distribute_state_pdaf such that fields in model are not overwritten. 
    
    Parameters
        ----------
        dim_p : int 
            Size of state_p. 
        state_p : 1D numpy array
            1D array containing the model fields for this process.
        
        Returns
        -------
        state_p :
            Same as intput state_p
    """
    return state_p

@dataclass
class Analyzer(ABC):
    """
    Abstract class defining DA filter/smoother. 
    
    Attributes
    ----------
    pe : parallelization.ProcessControl
        Object that controls parallelization in the filter. 
    covariance : covariances.Covariance 
        Object representing error covariance. 
    filtertype : int>0 
        Integer indicating type of filter. 
    subtype : int>=0
        Subtype
    prestep : list 
        List of functions to be applied just before DA. 
    poststep : list 
        List of functions to be applied just after DA.
    param_i : list of int 
        Filter parameters that are integers. 
    param_r : list of float 
        Filter parameters that are floats. 
    
    """
    pe : ProcessControl
    covariance : Covariance
    inflator : Inflator
    filtertype : FilterType
    subtype : int 
    presteps : list = field(default_factory=lambda:[])
    poststeps : list = field(default_factory=lambda:[])
    param_i : np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))
    param_r : np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    is_initialized : bool = False
        
    @abstractmethod 
    def build(self, das, model):
        """ 
        Add different user-defined functions to analyzer.
        
        Parameters 
        ----------
        das : dasystems.DAS object 
            Object containing all information about DA system. 
        model : list of models.Model object
            Different models using this comm_model. 
            
        """
       
    @check_init 
    def get_state(self, doexit, status_pdaf):
        """ Copy state from ensemble storage to input model. 
        
        Parameters 
        ----------
        doexit : bool 
            Flag indicating whether DA system needs to stop after current time step. 
        status_pdaf : options.PdafError 
            Status PDAF system. 
        
        Returns
        -------
        doexit :
            Flag indicating whether DA system needs to stop after current time step. 
        status_pdaf : 
            Status PDAF system. 
        
        """
        steps_forward = iNAN                
        return PDAF.get_state(steps_forward, doexit, self.u_next_observation, 
                              self.u_distribute_state, self.u_prepoststep, status_pdaf)
        
    @abstractmethod 
    @check_init
    def put_state(self, doexit, status_pdaf):
        """ 
        Copy output from model to ensemble storage. 
        
        Parameters 
        ----------
        doexit : bool 
            Flag indicating whether DA system needs to stop after current time step. 
        status_pdaf : options.PdafError 
            Status PDAF system. 
        
        Returns
        -------
        doexit :
            Flag indicating whether DA system needs to stop after current time step. 
        status_pdaf : 
            Status PDAF system. 
        
        """
        
    @check_init
    def u_prepoststep(self, step, dim_p, dim_ens, dim_ens_p, dim_obs_p, 
                      state_p, Uinv, ens_p, status_pdaf):
        """ 
        Pre- and postprocess ensemble. 
        
        Parameters 
        ----------
        step : int 
            Current time step. 
        dim_p : int 
            Size state vector at current process. 
        dim_ens : int 
            Number of ensemble members. 
        dim_ens_p : int 
            Number of ensemble members using this comm_model. 
        dim_obs_p : int 
            Size observation vector on this domain. 
        state_p : (dim_p,) ndarray 
            Best estimate state on this process. Not generally not initialized in the case of SEIK/EnKF/ETKF.
        Uinv : (dim_ens-1, dim_ens-1) ndarray 
            Inverse of matrix U containing mode coefficients. 
        ens_p : (dim_p, dim_ens) 
            Ensemble on this process. 
        status_pdaf : int 
            PDAF status. 
            
        Returns 
        -------
        state_p : 
            Best estimate state on this process. Not generally not initialized in the case of SEIK/EnKF/ETKF.
        Uinv : (dim_ens-1, dim_ens-1) ndarray 
            Inverse of matrix U containing mode coefficients. 
        ens_p : 
            Ensemble on this process. 
        status_pdaf :
            PDAF status. 
            
        """
        assert np.shape(state_p)==(dim_p,)
        assert np.shape(ens_p)==(dim_p,dim_ens_p)
        assert np.shape(Uinv)==(dim_ens-1,dim_ens-1)
        
        if step<0:
            for prestep in self.presteps:
                state_p, Uinv, ens_p, status_pdaf = prestep(step, state_p, Uinv, ens_p, status_pdaf)
        else:
            for poststep in self.poststeps:
                state_p, Uinv, ens_p, status_pdaf = poststep(step, state_p, Uinv, ens_p, status_pdaf)
                
        return state_p, Uinv, ens_p
        
class ETKF(Analyzer):
    """ Ensemble Transform Kalman filter. """
    
    def __init__(self, pe, covariance, inflator, presteps=[], poststeps=[]):
        super().__init__(pe, covariance, inflator,
                         FilterType.etkf, 0, presteps=presteps, poststeps=poststeps)
        
    @check_init
    def put_state(self, doexit, status_pdaf):
        return PDAF.put_state_etkf(self.u_collect_state, self.u_init_dim_obs,
                                   self.u_obs_op, self.u_init_obs, self.u_prepoststep, 
                                   self.u_prodRinvA, self.u_init_obsvar)
 
    def build(self, das, model):
        builder = das.builder
        self.u_collect_state = builder.build_collect_state(das, model)
        self.u_distribute_state = builder.build_distribute_state_init(das, model)
        self.u_init_dim_obs = builder.build_init_dim_obs(das, model)
        self.u_obs_op = builder.build_obs_op(das, model)
        self.u_init_obs = builder.build_init_obs(das, model)
        self.u_prodRinvA = builder.build_prodRinvA(das, model)
        self.u_init_obsvar = builder.build_init_obsvar(das, model)
        self.u_next_observation = builder.build_next_observation(das, model)  
        self.is_initialized = True
        
class AnalyzerBuilder:
    """ 
    Object to create a DA analyzer. 
    
    Methods 
    -------
    construct_analyzer 
        Create new analyzer and store in builder.
    get_analyzer 
        Collect the analyzer stored in builder.
    build
        Add different user-defined functions to analyzer.
    """
            
    def build_collect_state(self, das, model):
        return model.collect_state_pdaf
        
    def build_distribute_state(self, das, model):
        return model.distribute_state_pdaf 
    
    def build_distribute_state_init(self, das, model):
        #Don't overwrite initial conditions!
        if model.step == model.step_init:
            distribute_state_pdaf = identity_state_pdaf
        else:
            distribute_state_pdaf = model.distribute_state_pdaf
        return distribute_state_pdaf
    
    def build_next_observation(self, das, model):
        """ Build function u_next_observation_pdaf. """
        
        def u_next_observation_pdaf(step_now, nsteps, doexit, time_now):
            """ 
            Calculate time to next DA step. 
    
            Parameters
            ----------

            step_now : int
                Current time step.
            nsteps : int
                steps between assimilation. 
            doexit : int
                Whether exit PDAF assimilation.
            time_now : double
                Current model time. 
            
            Returns
            -------
            nsteps : int
                Steps to next observation time. 
            doexit : int
                Whether exit PDAF assimilation.
            time : double
                Current model time. 
            
            """
            #Steps to end simulation. 
            nsteps = das.final_step - step_now
            
            #Find the number of steps to next observation. 
            time_now = model.step2time(step_now)
            next_times = [obs.next_obs_time(time_now) for obs in das.observations if obs.next_obs_time(time_now) is not None]
            
            if len(next_times)==0:
                nsteps = 99999999
            else:
                next_time = np.min(next_times)
                nsteps = model.time2step(next_time) - step_now
                #doexit = model.time2step(next_time) > das.final_step
        
            #nsteps = nsteps if doexit else model.time2step(next_time)-step_now
            doexit = False
            doexit = int(doexit)
            
            return nsteps, doexit, time_now
        
        return u_next_observation_pdaf
    
    def build_init_dim_obs(self, das, model):
        
        def u_init_dim_obs_pdafomi(step, dim_obs):
            """
            Get size of observation vector. 
        
            Parameters 
            ----------
            step : int>=0 
                Current time step. 
            dim_obs : int>=0 
                Size of observation vector. 
            
            Returns 
            -------
            dim_obs : 
                Size of observation vector. 
            
            """ 
            time_now = model.step2time(step)
            
            for n,obs in enumerate(das.observations):
                obs.init(n+1, time_now)
            
            dim_obs = np.sum([obs.dim_obs for obs in das.observations if obs.doassim])
            return dim_obs
        
        return u_init_dim_obs_pdafomi
    
    def build_obs_op(self, das, model):
        
        def u_obs_op_pdafomi(step, dim_p, dim_obs_p, state_p, ostate):
            """
            Turn state vector to observation vector.
            
            Observation operators should have initialized at this time.

            Parameters
            ----------
            step : int
                Current time step.
            state_p : ndarray
                State vector on this process.
            ostate : ndarray
                State vector in observation space.
            
            Returns
            -------
            ostate : 
                Observation vector. 
            
            """
            assert len(state_p)==dim_p
            assert len(ostate)==dim_obs_p
            time_now = model.step2time(step)
        
            s = slice(0,0)
            for obs in das.observations:
                s = slice(s.stop, s.stop+obs.dim_obs)
                ostate = obs.obs_op(state_p, ostate)
                
            return ostate
        
        return u_obs_op_pdafomi
    
    def build_init_obs(self, das, model):
        
        def u_init_obs_pdafomi(step, dim_obs_p, obs_p):
            """ 
            It has to provide the vector of observations in observation_p for the current time step. 
            """
            obs_p = np.array([], dtype=float)
            for obs in das.observations:
                obs_p = np.append(obs_p, obs.observed)
            assert len(obs_p)==dim_obs_p
            return np.asfortranarray(obs_p, dtype=float)
        
        return u_init_obs_pdafomi
            
    def build_prodRinvA(self, das, model):
        
        def u_prodRinvA_pdaf(step, dim_obs_p, dim_ens, obs_p, A_p, C_p=None):
            """
            In the algorithms the product of the inverse of the observation error covariance 
            matrix with some matrix has to be computed.
            
            The interface has a difference for SEIK and ETKF: For ETKF the third argument is the ensemble
            size (dim_ens), while for SEIK it is the rank of the covariance matrix (usually ensemble size minus one).
            In addition, the second dimension of A_p and C_p has size dim_ens for ETKF, while it is rank for the SEIK filter. 
            (Practically, one can usually ignore this difference as the fourth argument of the interface can be named 
            arbitrarily in the routine.) 
            
            Parameters
            ----------
            step : int 
                Current time step 
            dim_obs_p : int 
                Size observation vector on this process. 
            dim_ens : int 
                Number of ensemble members. 
            obs_p : (dim_obs_p,) ndarray 
                Observed values on this process. 
            A_p : (dim_obs_p, dim_ens) ndarray 
                Input matrix. 
            C_p : (dim_obs_p, dim_ens) ndarray 
                Output matrix. 
                
            Returns
            -------
            C_p : (dim_obs_p, dim_ens) ndarray 
                Output matrix. 
                
            """
            if C_p is None:
                C_p = np.ones((dim_obs_p,np.size(A_p,1)), dtype=float, order='F') * fNAN
            
            s = slice(0,0)
            for obs in das.observations:
                s = slice(s.stop, s.stop+obs.dim_obs)
                Rinv = obs.covariance.inverse()
                C_p[s,:] = Rinv.left_multiply(A_p[s,:])
    
            return C_p
        
        return u_prodRinvA_pdaf
            
    def build_init_obsvar(self, das, model):
        
        def u_init_obsvar_pdaf(step, dim_obs_p, obs_p, meanvar):
            """
            The routine is called in the global filters during the analysis or by the routine that 
            computes an adaptive forgetting factor (PDAF_set_forget). The routine has to initialize
            the mean observation error variance. For the global filters this should be the global mean.
            
            Parameters
            ----------
            step : int 
                Current time step. 
            dim_obs_p : int 
                Size observation vector on this process. 
            obs_p : (dim_obs_p,) ndarray 
                Observed values on this process. 
            meanvar : float 
                Mean observation error variance. 
                
            Returns
            -------
            meanvar : float 
                Mean observation error variance. 
        
            """
            n_var, var = 0, 0.0
            for obs in das.observations:
                n_var += obs.covariance.pe.sum(len(obs.covariance.variance))
                var += obs.covariance.pe.sum(np.sum(obs.covariance.variance))
            
            return var/float(n_var)
            
        return u_init_obsvar_pdaf

