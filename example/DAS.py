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
import pyPDAF.PDAF as PDAF
import numpy as np
from options import ScreenOutput  

class DAS:
    """
    This class builds the DA system by composition of objects 
    listed below and contains a method to run the DA system
    forward in time. 
    
    Methods
    -------
    forward : 
        Run DA system forward to next time window.    

    Attributes
    ----------
    pe : `parallelization.parallelization`
        parallelization object
    model : `Model.Model`
        model object
    observations : list of `observations.OBS` objects
        observation object
    analyzer : `analyzer.Analyzer` object 
            Objects containing information about DA method. 
    is_initialized : bool 
        flag indicating whether DA system is ready for forward run
    dim_ens : int 
            Number of ensemble members.
        
    """

    def __init__(self, process_control, models, observations, 
                 analyzer):
        """
        Construct the DA system.. 

        Parameters
        ----------
        process_control : `parallelization.Parallelization` object
            Object controlling the use of multiple processes.
        models : list of `model.Model` objects
            Objects co
        observations : list of `observations.Obs` objects
            Objects containing observations and sample operators. 
        analyzer : `analyzer.Analyzer` object 
            Objects containing information about DA method. 
            
        """
        self.pe = process_control
        self.models = models
        self.obs = observations 
        self.analyzer = analyzer
        self.is_initialized = False 
        self.dim_ens = len(self.models)

    def init(self, verbose=ScreenOutput.quiet):
        """
        Setup Fortran backend and model for assimilation. 
        
        Parameters
        ----------
        verbose : options.ScreenOuput object 
            Option for screen output. 
        """
        # init models
        for model in self.models:
            model.init_fields()

        # init observations
        PDAF.omi_init(len(self.obs))
        
        # init PDAF 
        status = PDAF.init(self.analyzer.filtertype,
                           self.analyzer.subtype,
                           0,
                           self.analyzer.param_i,
                           self.analyzer.param_r,
                           self.pe.COMM_model.py2f(),
                           self.pe.COMM_filter.py2f(),
                           self.pe.COMM_couple.py2f(), self.assimilatepe.task_id,
                           self.pe.n_modeltasks, self.pe.filterpe,
                           self.u_init_ens_pdaf, int(self.verbose),
                           )
        
        # Mark initialization as complete. 
        self.is_initialized = True
        
    def forward(self, step, verbose=ScreenOutput.quiet):
        """
        Step the DA system forward in time to 
        the next assimilation time. 

        Parameters
        ----------
        step : int
            Current time step.
        verbose : options.ScreenOuput object 
            Option for screen output. 
            
        """
        # Check if the Fortran backend and model are ready. 
        if not self.is_initialized:
            self.init()
        #Execute predicition step of DA.
        self.model.step(step)
        #Execute the update step. 
        self.analyzer.assimilate(verbose)
        
    def u_unit_ens_pdaf(self, filtertype, dim_p, dim_ens, state_p, uinv, ens_p, status_pdaf):
        """
        Allocate memory for ensemble members and initialize the different members. 

        Parameters
        ----------
        filtertype : int
            Type of filter.
        dim_p : int 
            Size of state vector on this process.
        dim_ens : int 
            Number of ensemble members. 
        state_p : ndarray of shape (dim_p,)
            1D state vector on local process.
        uinv : ndarray of shape (dim_ens-1,dim_ens-1)
            Inverse of U with U by SVD covariance P=VUVt
        ens_p : ndarray of shape (dim_p,dim_ens)
            Ensemble state vector on local process. 
        status_pdaf : int
            Status of PDAF.

        Returns
        -------
        uinv : ndarray of shape (dim_ens-1,dim_ens-1)
            Inverse of U with U by SVD covariance P=VUVt
        ens_p : ndarray of shape (dim_p,dim_ens)
            Ensemble state vector on local process. 
        status_pdaf : int
            Status of PDAF.
            
        """
        #Get initial conditions for each ensemble member.
        ens_p = np.empty(shape=(dim_p, dim_ens), order='F')
        for n,model in enumerate(self.models):
            ens_p[:,n] = model.collect_state_pdaf(dim_p, ens_p[:,n])
        
        #Get SVD of covariance matrix. Only relevant for SEEK. 
        uinv = self.analyzer.get_uinv(dim_ens)
        
        return state_p, uinv, ens_p, status_pdaf
        
    # def assimilate(self):
    #     """ 
    #     Update model fields using the DA correction. 
    #     """
        
        
    #     # Call the Fortran backend to carry out the assimilation.  
    #     status = PDAF.omi_assimilate_global(self.model.collect_state_pdaf,
    #                                         self.model.distribute_state_pdaf,
    #                                         U_init_dim_obs_PDAFomi,
    #                                         U_obs_op_PDAFomi,
    #                                         U_prepoststep_ens_pdaf,
    #                                         U_next_observation_pdaf)
        
    #     # Have process control deal with the errors. 
    #     if status!=0:
    #         msg = "PDAF.omi_assimilate_global failed."
    #         self.pe.raise(msg)
        
        
        
    #     if USE_PDAF:
    #         PDAF_caller.assimilate_pdaf(self.model, self.obs, self.pe,
    #                                     self.assim_dim, self.localization,
    #                                     self.filter_options.filtertype)
