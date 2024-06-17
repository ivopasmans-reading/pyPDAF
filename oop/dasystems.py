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
from options import ScreenOutput, PdafError, fNAN, iNAN

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
    analyzer : `analyzers.Analyzer` object 
            Objects containing information about DA method. 
    is_initialized : bool 
        flag indicating whether DA system is ready for forward run
    dim_ens : int 
            Number of ensemble members.
        
    """

    def __init__(self, process_control, models, observations, 
                analyzer_builder):
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
        analyzer_builder : `analyzers.AnalyzerBuilder` object 
            Objects containing information about DA method. 
            
        """
        self.pe = process_control
        self.models = models
        self.obs = observations 
        self.builder = analyzer_builder
        self.is_initialized = False 
        self.dim_ens = len(self.models)
        
    def run(self, steps, vebose=ScreenOutput.quiet):
        """ 
        Run the ensemble for steps assimilating on the way. 
        
        Parameters
        ----------
        steps : int 
            Length of DA system run in time steps. 
            
        """
        steps_left = steps 
        doexit = False 
        
        #Initialize models and setup DA 
        self.init(verbose=verbose)
        
        #Create DA method 
        last_step = self.step_init + steps
        self.builder.construct_analyzer()
        
        #Run DA forward in time. 
        status_pdaf = int(PdafError.none)
        while not bool(doexit):
            #Run model forward to next DA moment.
            for model in self.models_p:
                self.builder.build(model, self.obs, last_step)
                analyzer = self.builder.get_analyzer()
                
                #Copy state from ensemble storage to input model.
                steps_forward, time_now, doexit, status_pdaf  = analyzer.get_state(doexit, status_pdaf)
                if status_pdaf!=int(PdafError.none):
                    self.pe.comm_world.throw_error(f"Get_state critically failed to copy ensemble member at step {step}.")   
                    
                #Run model forward in time. 
                model.step(step, steps_forward)
                    
                #Copy output from model to ensemble storage.
                status_pdaf = self.analyzer.put_state(doexit, status_pdaf)
                if status_pdaf!=int(PdafError.none):
                    self.pe.comm_world.throw_error(f"Put_state critically failed to copy ensemble member at step {step}.")  
                    
            #Update steps left. 
            steps_left = steps_left - steps_forward                

    def init(self, verbose=ScreenOutput.quiet):
        """
        Setup Fortran backend and model for assimilation. 
        
        Parameters
        ----------
        verbose : options.ScreenOuput object 
            Option for screen output. 
        """
        #Only initialize models on using this comm_model. 
        self.models_p = [model.init_field(self.pe.comm_model) for n,model in enumerate(self.models) if self.pe.for_comm_model(n)]
        if len(self.models_p)==0:
            self.pe.comm_world.throw_error(f"No members assigned to communicator {self.pe.comm_model.name}")
        if not np.all([model.step==self.models_p[0].step for model in self.models_p]):
            self.pe.comm_world.throw_error(f"All models must start and same time step.")
        else:
            self.step_init = self.models_p[0].step + 0

        # init observations
        PDAF.omi_init(len(self.obs))
        
        # init PDAF 
        status_pdaf = PDAF.init(self.analyzer.filtertype,
                                self.analyzer.subtype,
                                self.step_init,
                                self.analyzer.param_i,
                                self.analyzer.param_r,
                                self.pe.comm_model.py2f(),
                                self.pe.comm_filter.py2f(),
                                self.pe.comm_couple.py2f(), 
                                self.pe.this_comm_model+1,
                                self.pe.n_comm_models, 
                                int(self.pe.this_filter_model is not None),
                                self.u_init_ens_pdaf, int(self.verbose),
                                )
        #Error handling
        if status_pdaf!=int(PdafError.none):
            self.pe.comm_model.throw_error("PDAF.init critically failed to initialized ensemble.")            
            
        # Mark initialization as complete. 
        self.is_initialized = True        
        
    def u_init_ens_pdaf(self, filtertype, dim_p, dim_ens, state_p, uinv, ens_p, status_pdaf):
        """
        Allocate memory for ensemble members and initialize the different members. 
        
        This method is only run on processes part of self.pe.comm_filter. 

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
        #Allocate memory. 
        ens_p = np.ones((dim_p, dim_ens), dtype=float, order='F') * fNAN
        state_p = np.ones((dim_p,), dtype=float, order='F') * fNAN
        uinv = np.ones((dim_ens-1,dim_ens-1), dtype=float, order='F') * fNAN
        
        #TODO: implement initialization for mode filters. 
        
        return state_p, uinv, ens_p, status_pdaf