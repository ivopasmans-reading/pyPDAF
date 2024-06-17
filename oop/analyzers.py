from abc import ABC, abstractmethod
from dataclasses import dataclass 
from options import fNAN, iNAN, FilterType
import pyPDAF.PDAF as PDAF
import numpy as np

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
    filtertype : int>0 
        Integer indicating type of filter. 
    subtype : int>=0
        Subtype
    param_i : list of int 
        Filter parameters that are integers. 
    param_r : list of float 
        Filter parameters that are floats. 
    
    """
    filtertype : FilterType
    subtype : int 
    param_i : list[int]
    param_r : list[float]
    
    @abstractmethod 
    def check_settings(self):
        """ Check whether the attributes are valid for class. """
        
    @abstractmethod 
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
        
    @abstractmethod 
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
        if step<0:
            return self.u_prestep(step, dim_p, dim_ens, dim_ens_p, dim_obs_p, 
                                  state_p, Uinv, ens_p, status_pdaf)
        else:
            return self.u_poststep(step, dim_p, dim_ens, dim_ens_p, dim_obs_p, 
                                   state_p, Uinv, ens_p, status_pdaf)
        
class EnsembleGlobalFilter(Analyzer):
    """ Ensemble Kalman filters. """
    
    def check_settings(self):
        #TODO: implement checks. 
        pass
        
    def get_state(self, doexit, status_pdaf):
        steps_forward, time_now = iNAN, fNAN                  
        return PDAF.get_state(steps_forward, time_now, doexit, self.u_next_observation_pdaf, 
                              self.u_distribute_state_pdaf, self.u_prepoststep, status_pdaf)
        
    def put_state(self, doexit, status_pdaf):
        return PDAF.put_state_global(self.u_collect_state_pdaf, self.u_init_dim_obs,
                                     self.u_obs_op, self.u_prepoststep, status_pdaf)
        
class AnalyzerBuilder(ABC):
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
    
    @abstractmethod 
    def construct_analyzer(self):
        """ Create new analyzer and store in builder. """
            
    def get_analyzer(self):
        """" Collect the analyzer stored in builder. """
        return self.analyzer
    
    def build(self, model, observations, final_step):
        """ 
        Add different user-defined functions to analyzer.
        
        Parameters 
        ----------
        model : list of models.Model object
            Different models using this comm_model. 
        observations : list of observations.Obs objects
            Observation operators. 
        final_step : int 
            Last model step to be taken by DA system. 
            
        """
        self.analyzer.u_prestep = self.build_prestep(model, observations, final_step)
        self.analyzer.u_poststep = self.build_poststep(model, observations, final_step)
        
        self.analyzer.u_next_observation_pdaf = self.build_next_observation_pdaf(model, observations, final_step)
        self.analyzer.u_dim_obs_pdafomi = self.build_dim_obs_pdafomi(model, observations, final_step)
        self.analyzer.obs_op_pdafomi = self.build_obs_op_pdafomi(model, observations, final_step)
        
    def build_prestep(self, model, observations, steps_left):
        """ Build u_prestep. """
        
        def u_prestep(step, dim_p, dim_ens, dim_ens_p, dim_obs_p, 
                      state_p, Uinv, ens_p, status_pdaf):
            """ See u_prepoststep. """
            return state_p, Uinv, ens_p, status_pdaf
        
        return u_prestep
        
    def build_poststep(self, model, observations, steps_left):
        """ Build u_poststep. """
        
        def u_poststep(step, dim_p, dim_ens, dim_ens_p, dim_obs_p, 
                       state_p, Uinv, ens_p, status_pdaf):
            """ See u_prepoststep """
            return state_p, Uinv, ens_p, status_pdaf
        
        return u_poststep
        
    def build_collect_state_pdaf(self, model, observations, steps_left):
        return model.collect_state_pdaf
        
    def build_distribute_state_pdaf(self, model, observations, steps_left):
        #Don't overwrite initial conditions!
        if model.step == model.step_init:
            distribute_state_pdaf = identity_state_pdaf
        else:
            distribute_state_pdaf = model.distribute_state_pdaf
        return distribute_state_pdaf
    
    def build_next_observation_pdaf(self, model, observations, final_step):
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
            #Maximum number of steps left 
            nsteps = final_step - step_now 
        
            #Find the number of steps to next observation. 
            time_now = model.step2time(step_now)
            next_times = [obs.next_obs_time(time_now) for obs in observations if obs.next_obs_time(time_now) is not None]
            if len(next_times)>0:
                next_time = np.min(next_times)
                nsteps = min(nsteps, model.time2step(next_time) - step_now)
            
            doexit = bool(doexit) or nsteps <= 0
            doexit = int(doexit)
        
            return nsteps, doexit, time_now
        
        return u_next_observation_pdaf
    
    def build_dim_obs_pdafomi(self, model, observations, steps_left):
        
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
            
            for n,obs in enumerate(observations):
                obs.init(n+1, time_now)
            
            return np.sum([obs.dim_obs for obs in observations if obs.doassim])
        
        return u_init_dim_obs_pdafomi
    
    def build_obs_op_pdafomi(self, model, observations, steps_left):
        
        def u_obs_op_pdafomi(step, dim_p, dim_obs_p, state_p, ostate):
            """
            Turn state vector to observation vector. 

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
            time_now = model.step2time(step)
        
            for obs in observations:
                ostate = obs.obs_op(time_now, state_p)
            
            return ostate
        
        return u_obs_op_pdafomi
    
class EnsembleGlobalBuilder(AnalyzerBuilder):
    """ 
    Builder to create one of the global ensemble filters. 
    
    Attributes
    ----------
    filtertype : int>0 
        Integer indicating type of filter. 
    subtype : int>=0
        Subtype
    param_i : list of int 
        Filter parameters that are integers. 
    param_r : list of float 
        Filter parameters that are floats. 

    """
    
    def __init__(self, filtertype, subtype, param_i, param_r):
        self.filtertype = filtertype 
        self.subtype = subtype 
        self.param_i = param_i 
        self.param_r = param_r
    
    def construct_analyzer(self):
        self.analyzer = EnsembleGlobalFilter(self.filtertype, self.subtype, self.param_i, self.param_r)
    
    def build(self, model, observations, final_step):
        self.super().build(model, observations, final_step)
        self.analyzer.u_collect_state_pdaf = self.build_collect_state_pdaf(model, observations, final_step)
        self.analyzer.u_distribute_state_pdaf = self.build_distribute_state_pdaf(model, observations, final_step)
        
    
    # def build_dim_obs_l_pdafomi(self, model, observations, steps_left):
        
    #     def init_dim_obs_l_pdafomi(domain_p, step, dim_obs, dim_obs_l):
    #         """
    #         Initialise local observation dimension.

    #         Parameters
    #         ----------
    #         domain_p : int
    #             Index of current local analysis domain.
    #         step : int
    #             Current time step.
    #         dim_obs : int
    #             Dimension of observation vector.
    #         dim_obs_l : int
    #             Dimension of local observation vector.
            
    #         Returns 
    #         -------
    #         dim_obs : 
    #             Size of observation vector. 
    #         dim_obs_l : int 
    #             Dimension of local observation vector. 
            
    #         """
    #         time_now = model.step2time(step)
    #         dim_obs = np.sum([obs.dim_obs(time_now) for obs in observations])
    #         dim_obs_l = np.sum([obs.(time_now, domain_p) for obs in observations])
    #         return dim_obs, dim_obs_l
        
    #     return init_dim_obs_l_pdafomi