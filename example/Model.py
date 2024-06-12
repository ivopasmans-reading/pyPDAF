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
import copy
from abc import ABC, abstractmethod

import model.shift

class Model(ABC):
    """ 
    Abstract class for a PDAF compatible model. 
    
    Methods
    -------
    init_field 
        Initialises model field at beginning of model run. 
    step 
        Step the model forward in time from given step.  
    collect_state_pdaf
        Outputs local model fields as array to be collected by PDAF.
    distribute_state_pdaf
        Accepts corrected local model fields from PDAF as 1D array. 
        
    """
    
    @abstractmethod
    def init_field(self):
        """
        Initialises model field at beginning of model run. 
        """
    
    @abstractmethod 
    def step(self, step):
        """ 
        Step the model forward in time from given step.  
        
        Parameters
        ----------
        step : int
            current time step
        use_pdaf : bool
            whether PDAF is used at this step
            
        """
        
    @abstractmethod
    def collect_state_pdaf(self, dim_p, state_p):
        """Outputs local model fields as array to be collected by PDAF.

        Aim of this method is to reshape the different models fields 
        into 1 long 1D array that and returns this one so it can 
        be processed by PDAF. As such it is the opposite of 
        `distribute_state_pdaf`. Relies on Python's pass by reference. 
        The interface of this method should not be changed as it must 
        match the equivalent PDAF interface. 

        Parameters
        ----------
        dim_p : int 
            Size of array state_p.
        state_p : ndarray
            Allocated 1D array to store the output of this method.    
            
        Returns
        -------
        state_p : 1D numpy array 
            Model fields for this process concatenated into 1D array 
            in Fortran ordering.  
    
    """
    
    @abstractmethod
    def distribute_state_pdaf(self, dim_p, state_p):
        """Accepts corrected local model fields from PDAF as 1D array. 

        Aim of this method is to reshape the 1D array from PDAF back
        into different model fields of different sizes. As such 
        it is the opposite to `collect_state_pdaf`. Relies on Python's 
        pass by reference. The interface of this method should not be 
        changed as it must match the equivalent PDAF interface. 

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
        
class Ar1Model(Model):
    
    def __init__(self, nx, nt, comm_controller, 
                 parameters=np.array([1.0])):
        self.set_attributes(nt, nx, nx)
        self.comm = comm_controller 
        self.ar = self.parameters
        
    def init_field(self):
        self.field_p = np.random.normal(size=np.shape(self.field_p), scale=self.ar[0])
        for n in range(1, np.size(self.field_p,0)):
            self.field_p[n] = ar[1] * self.field_p[n-1] + np.sqrt(1-ar[1]**2) * self.field_p[n]
    
    def step(self, step, use_pdaf):
        self.field = np.roll(self.field, 1, axis=0)
        return self.field
        
      


class OldModel:

    """Model information in PDAF

    Attributes
    ----------
    field_p : ndarray
        PE-local model field
    nx : ndarray
        integer array for grid size
    nx_p : ndarray
        integer array for PE-local grid size
    total_steps : int
        total number of time steps
    """

    def __init__(self, nx, nt, pe):
        """constructor

        Parameters
        ----------
        nx : ndarray
            integer array for grid size
        nt : int
            total number of time steps
        pe : `parallelization.parallelization`
            parallelization object
        """
        # model size
        self.nx = list(nx)
        # model size for each CPU
        self.get_nxp(pe)
        # model time steps
        self.total_steps = nt

    def get_nxp(self, pe):
        """Compute local-PE domain size/domain decomposition

        Parameters
        ----------
        pe : `parallelization.parallelization`
            parallelization object
        """
        self.nx_p = copy.copy(self.nx)

        try:
            assert self.nx[-1] % pe.npes_model == 0
            self.nx_p[-1] = self.nx[-1]//pe.npes_model
        except AssertionError:
            print((f'...ERROR: Invalid number of'
                   f'processes: {pe.npes_model}...'))
            pe.abort_parallel()

    def init_field(self, filename, mype_model):
        """initialise PE-local model field

        Parameters
        ----------
        filename : string
            input filename
        mype_model : int
            rank of the process in model communicator
        """
        # model field
        self.field_p = np.zeros(self.nx_p)
        offset = self.nx_p[-1]*mype_model
        self.field_p = np.loadtxt(
                                    filename
                                    )[:, offset:self.nx_p[-1] + offset]

    def step(self, pe, step, USE_PDAF):
        """step model forward

        Parameters
        ----------
        pe : `parallelization.parallelization`
            parallelization object
        step : int
            current time step
        USE_PDAF : bool
            whether PDAF is used
        """
        model.shift.step(self, pe, step, USE_PDAF)

    def printInfo(self, USE_PDAF, pe):
        """print model info

        Parameters
        ----------
        USE_PDAF : bool
            whether PDAF is used
        pe : `parallelization.parallelization`
            parallelization object
        """
        do_print = USE_PDAF and pe.mype_model == 0
        do_print = do_print or \
            (pe.task_id == 1 and pe.mype_model == 0 and not USE_PDAF)
        if do_print:
            print('MODEL-side: INITIALIZE PARALLELIZED Shifting model MODEL')
            print(f'Grid size: {self.nx}')
            print(f'Time steps {self.total_steps}')
            print(f'-- Domain decomposition over {pe.npes_model} PEs')
            print(f'-- local domain sizes (nx_p): {self.nx_p}')
