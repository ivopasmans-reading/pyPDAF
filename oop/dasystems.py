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
from general import ScreenOutput, PdafError, fNAN, iNAN, check_init

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
                analyzer, builder):
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
        analyzer : `analyzers.Analyzer` object 
            Objects containing information about DA method. 
        builder : analyzer.AnalyzerBuilder object 
            Object that builds the user defined PDAF subroutines. 
            
        """
        self.pe = process_control
        self.models = models
        self.observations = observations 
        self.analyzer = analyzer
        self.builder = builder 
        self.is_initialized = False 
        self.dim_ens = len(self.models)
        
    def run(self, steps, verbose=ScreenOutput.standard):
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
        
        #Determine the last step to be calculated by model. 
        self.final_step = self.step_init + steps
        
        #Run DA forward in time. 
        step_da, status_pdaf = self.step_init, int(PdafError.none)
        while step_da<=self.final_step:
            #Run model forward to next DA moment.
            for model in self.models_p:
                self.analyzer.build(self, model)
                
                #Copy state from ensemble storage to input model.
                #DONT use doexit here as otherwise to output from last analysis in not copied back in model. 
                steps_forward, _, _, status_pdaf = self.analyzer.get_state(doexit, status_pdaf)
                if status_pdaf!=int(PdafError.none):
                    self.pe.comm_world.throw_error(f"Get_state critically failed to copy ensemble member at step {step}.")  
                step_da = model.step + steps_forward
                
                #Run model forward in time. 
                if verbose==ScreenOutput.standard and self.pe.comm_model.is_main:
                    print(f"Running model {steps_forward} steps from time {model.time}.")
                if verbose==ScreenOutput.standard:
                    state_p = model.collect_state_pdaf(model.dim_state,None)
                    print('State min/max/mean at process {:d}: {:.3e} {:.3e} {:.3e}'.format(self.pe.comm_model.mype,
                                                                                np.min(state_p),np.max(state_p),
                                                                                np.mean(state_p)))
                model.step_forward(model.step, min(self.final_step - model.step, steps_forward))
                if verbose==ScreenOutput.standard:
                    state_p = model.collect_state_pdaf(model.dim_state,None)
                    print('State min/max/mean at process {:d}: {:.3e} {:.3e} {:.3e}'.format(self.pe.comm_model.mype,
                                                                                 np.min(state_p),np.max(state_p),
                                                                                 np.mean(state_p)))
                    
                #Following part has to be skipped if the last observation has been assimilated.
                if step_da > self.final_step:
                    continue
                    
                #Copy output from model to ensemble storage.
                status_pdaf = self.analyzer.put_state(doexit, status_pdaf)
                if status_pdaf!=int(PdafError.none):
                    self.pe.comm_world.throw_error(f"Put_state critically failed to copy ensemble member at step {step}.")  
                    
                #Deallocate observations after DA. 
                for obs in self.observations:
                    obs.deallocate()       

    def init(self, verbose=ScreenOutput.quiet):
        """
        Setup Fortran backend and model for assimilation. 
        
        Parameters
        ----------
        verbose : options.ScreenOuput object 
            Option for screen output. 
        """
        #Only initialize models on using this comm_model. 
        self.models_p = [model for n,model in enumerate(self.models) if self.pe.for_comm_model(n)]
        [model.init_fields(self.pe.comm_model) for model in self.models_p]
        if len(self.models_p)==0:
            self.pe.comm_world.throw_error(f"No members assigned to communicator {self.pe.comm_model.name}")
        if not np.all([model.step==self.models_p[0].step for model in self.models_p]):
            self.pe.comm_world.throw_error(f"All models must start and same time step.")
        else:
            self.step_init = self.models_p[0].step + 0

        # init observations
        PDAF.omi_init(len(self.observations))
        
        #Number of ensemble members. 
        self.n_members = len(self.models)
        #Number of ensemble members using this comm_model. 
        self.n_members_p = len(self.models_p)
        
        # init PDAF 
        param_i = np.append([self.models_p[0].dim_state_p, self.n_members], self.analyzer.param_i)
        param_r = np.append([self.analyzer.inflator.forgetting_factor], self.analyzer.param_r)
        _,_,status_pdaf = PDAF.init(int(self.analyzer.filtertype),
                                    int(self.analyzer.subtype),
                                    int(self.step_init),
                                    np.array(param_i, dtype=int, order='F'),
                                    np.array(param_r, dtype=float, order='F'),
                                    self.pe.comm_model.py2f(),
                                    self.pe.comm_filter.py2f(),
                                    self.pe.comm_couple.py2f(), 
                                    self.pe.this_comm_model+1,
                                    self.pe.n_comm_models, 
                                    int(self.pe.this_comm_filter is not None),
                                    self.u_init_ens_pdaf, int(verbose),
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