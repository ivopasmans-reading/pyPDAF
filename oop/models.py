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
from abc import ABC, abstractmethod
import numpy as np 
from general import check_init

class Model(ABC):
    """ 
    Abstract class for a PDAF compatible model. 
    
    Methods
    -------
    init_fields 
        Initialises model field at beginning of model run. 
    step_forward
        Step the model forward in time from given step.  
    collect_state_pdaf
        Outputs local model fields as array to be collected by PDAF.
    distribute_state_pdaf
        Accepts corrected local model fields from PDAF as 1D array.
    dim_state : int 
        Size of model state. 
    dim_state_p : int 
        Size of model state on local process.
    time : float 
        Current model time.  
        
    Attributes
    ----------
    dt : float>0
        Numerical time step.
    step : int>=0
        Current model step. 
    step_init : int>=0
        Step at which model is initialized. 
    time_init : float 
        Time at which model is initialized. 
    control : parallelization.ProcessControl 
        Object controlling parallelization. 
    
    """
    
    @property
    @abstractmethod
    def dim_state(self):
        """Return size of total model state."""
        
    @property
    def dim_state_p(self):
        """
        Return size of model state on this process.
        
        By default is assumed that only 1 process is used.
        """
        return self.dim_state
    
    @property 
    def time(self):
        """ Return current model time interval. """
        return self.step2time(self.step)
    
    def step2time(self, step):
        """ Return time for step.  """
        return float(self.dt * (step - self.step_init)) + self.time_init
    
    def time2step(self, time):
        """ Return step associated with time. """
        return int( (time - self.time_init) / self.dt )
    
    @abstractmethod
    @check_init
    def init_fields(self, process_control):
        """
        Initialises model field at beginning of model run. 
        
        After this method step_init and time_init must be set. 
        
        Parameters
        ----------
        process_control : parallelization.ProcessControl 
            Object containing information about processes used by this model. 
        """
    
    @abstractmethod 
    def step_forward(self, step, steps_forward):
        """ 
        Step the model forward in time from given step.  
        
        Parameters
        ----------
        step : int>=0
            Current time step.
        steps_forward : int>0
            Number of steps        
            
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
    """ 
    Example of a very simple model. 
    
    Attributes 
    ----------
    dt : float 
        Time step
    shape : (2,) tuplet of int
        Size of the model field. 
    var : float 
        Variance AR1 process. 
    corr : float 
        Correlation AR1 process spaced 1 spatial step apart. 
    seed : int 
        Seed for random number generation. 
    is_initialized : bool
        Flag indicating whether fields were created. 
        
    Methods 
    -------
    nn_interpolate
        Given grid index coordinates return 
        indices in state_p/weights to carry out 
        nearest neighbor interpolation. 
        
    """
    
    def __init__(self, dt, shape, var, corr, seed=None, save_steps=[]):
        self.dt = dt 
        self.shape = shape 
        self.par = (np.sqrt(var * (1-corr**2)), corr)
        
        self.save_steps = np.array(save_steps, dtype=int)
        self.saved_output = np.empty((len(save_steps),)+tuple(self.shape), dtype=float)
        
        if seed is not None:
            np.random.seed(seed)
        self.random_state = np.random.get_state()
       
    @property 
    def dim_state(self):
        return np.prod(self.shape)
        
    def init_fields(self, process_control):
        np.random.set_state(self.random_state)
        
        self.control = process_control
        self.step_init, self.time_init, self.step = 0, 0.0, 0
        self.values = np.empty(self.shape, dtype=float)
        
        self.values[0] = np.random.normal(size=(1,self.shape[1]))
        for n in range(1,np.size(self.values,1)):
            self.values[0,n] = self.par[1] * self.values[0,n-1] + self.par[0] * self.values[0,n]
        for n in range(np.size(self.values,0)):
            self.values[n] = np.roll(self.values[0], n)
            
        self.save_output(self.step_init)
        self.random_state = np.random.get_state()
        self.is_initialized = True
      
    @check_init      
    def step_forward(self, step, steps_forward):   
        np.random.set_state(self.random_state)
        
        for _ in range(steps_forward):
            self.values[0] = np.roll(self.values[0], -1) 
            self.values[0,-1]  = self.par[0] * np.random.normal()
            self.values[0,-1] += self.par[1] * self.values[0,-2]
            
        for n in range(np.size(self.values,0)):
            self.values[n] = np.roll(self.values[0], n)
            
        self.step += steps_forward
        self.save_output(self.step)
        
        self.random_state = np.random.get_state()
        
    def collect_state_pdaf(self, dim_p, state_p):
        state_p = np.reshape(self.values, (-1,))
        return state_p
        
    def distribute_state_pdaf(self, dim_p, state_p):
        self.values = np.reshape(state_p, self.shape)
        return state_p
    
    def nn_interpolator(self, coords):
        coords  = np.array(np.round(coords), dtype=int)
        indices = np.ravel_multi_index(coords.T, dims=self.shape)
        indices = np.reshape(indices, (-1,1))
        weights = np.ones_like(indices, dtype=float)
        weights = np.where(np.logical_and(indices>=0, indices<self.dim_state),
                           weights, 0.0)
        return indices, weights
    
    def save_output(self, step):
        matches = np.where(step==self.save_steps)[0]
        for n in matches:
            self.saved_output[n] = self.values
        
        
