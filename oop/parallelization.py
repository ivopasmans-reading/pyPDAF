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
from mpi4py import MPI
import numpy as np
import sys

class ProcessControl:
    """
    Handles parallelization to be handled in a generic way in the code. 
    
    This class contains several methods that the codes needs to 
    run in parallel but in such a way that they are available even
    if the parallellization scheme doesn't supply them. By default
    serial run is assumed. 
    """

    def __init__(self, name="world", parent=None):
        self.name = name
        self.parent = None
        
    @property 
    def mype(self):
        """ Return integer identifying current process."""  
        return 0 

    @property 
    def npes(self):
        """ Return total number of processes in this communicator."""
        return 1 
    
    @property 
    def mythread(self):
        """ Return integer identifying thread of current process."""
        return 0 
    
    @property 
    def nthreads(self):
        """ Return total number of threads on this process."""
        return 1 
        
    def init(self):
        """Initialize parallization. Call at beginning code."""
        pass
        
    def finalize(self):
        """End parallization. Call at end code."""
        pass
    
    def abort(self):
        """ Force end run after error."""
        sys.exit(1)
    
    def create_subcomm(self, names):
        """ 
        Create new communicators from subset of processes. 
        
        names : dict
            Dictionary with name of subcommunicator as keys and 
            processes for that subcomm as values. 
        
        """
        keys = [key for key,value in names.items() if self.mype in value]
        if len(keys)<=1:
            return self
        else:
            raise ValueError("Process part of multiple subcommunicators.")
        
    @property 
    def is_main(self):
        """ 
        Indicate whether this process should be considered 
        the main one for IO, gather, etc."""
        return self.mype==0
    
    def py2f(self):
        """ Convert comm into integer for Fortran use. """
        return 0
    
    def throw_error(self, message):
        """
        Stop execution because of error. 
        
        message : str 
            Error message.
        """
        print("------------------------------------------------------")
        print(f"Critical error on process {self.mype}, thread {self.mythread} of {self.name}.")
        print(message)
        print("------------------------------------------------------")
        
        if self.parent is None:
            self.abort()
        else:
            self.parent.throw_error(message)
            
    def sum(self, send):
        """ Sum over all processes in communicator."""
        return send
    
class MpiControl(ProcessControl):
    """ Communicator using created using mpi4py. """
    
    def __init__(self, name="world", parent=None, comm=None):
        """ Constructor. Default comm is mpi_comm_world."""
        super().__init__(name, parent)
        if comm is None:
            self.comm = MPI.COMM_WORLD 
        else:
            self.comm = comm 
            
        #Add name to comm. 
        self.comm.Set_name(self.name)
            
    @property 
    def mype(self):
        """ Return integer identifying current process."""  
        return self.comm.Get_rank()

    @property 
    def npes(self):
        """ Return total number of processes in this communicator."""
        return self.comm.Get_size()
        
    def init(self):
        """Initialize parallization. Call at beginning code."""
        if self.comm is MPI.COMM_WORLD and not MPI.Is_initialized():
            MPI.Init()
        
    def finalize(self):
        """End parallization. Call at end code."""
        if self.comm is MPI.COMM_WORLD and MPI.Is_initialized():
            self.comm.Barrier()
            MPI.Finalize()
        else:
            raise NotImplemented("Only MPI.COMM_WORLD can be initialized.")
    
    def abort(self):
        """ Force end run after error."""
        self.comm.Abort(1)
    
    def create_subcomm(self, names):
        """ 
        Create new communicators from subset of processes. 
        
        names : dict
            Dictionary with name of subcommunicator as keys and 
            processes for that subcomm as values. 
        
        """
        keys = [(n+1,key) for n,(key,value) in enumerate(names.items()) if self.mype in value]
        if len(keys)==0:
            comm = self.comm.Split(MPI.UNDEFINED, self.mype)
            new  = MpiControl(name="sub_"+self.name+"_undefined", comm=comm, parent=self)
        elif len(keys)==1:
            color, name = keys[0]
            comm = self.comm.Split(color, self.mype)
            new  = MpiControl(comm=comm, name=name, parent=self)
        else:
            raise ValueError("Process can only be part of a single subcommunicator.")
        
        return new
        
    def py2f(self):
        return self.comm.py2f()
    
    def sum(self, send):
        recv = 0 
        recv = MPI.reduce(send, recv, MPI.sum)
        return recv
        
class PdafParallelization:
    """ 
    This class holds the three communicators used by PDAF.
    
    Attributes
    ----------
    comm_world : `ProcessControl` object 
        Overall process controller. 
    comm_model : `ProcessControl` object 
        Process controller to be used by model. 
    comm_couple : `ProcessControl` object 
        Process controller for communication between ensemble members. 
    comm_filter : `ProcessControl` object
        Controller containing process used by DA calculations. 
    pe_matrix : 2D ndarray
        Array with each row containing processes for 1 comm_model and 
        each column for 1 comm_couple. 
    
    Methods
    -------
    for_comm_model :
        Indicates whether ensemble member uses comm_model to run forward.

    """
    
    def __init__(self, npes_per_model=1):
        """
        Constructor. 
        
        Parameters
        ----------
        npes_per_model : int>0 
            Number of processes to be used for each model run. 
            
        """
        self.npes_per_model = int(npes_per_model)
        
    def initialize(self):
        """ Initialize the processes."""
        #Start processes
        self.comm_world = MpiControl()
        self.comm_world.init()
        
        #Make matrix of which processes should end in which comms
        self._create_pe_matrix()
        
        #For each row create comm_model 
        split = dict([(f"comm_model{n:03d}",row) for n,row in enumerate(self.pe_matrix)])
        self.comm_model = self.comm_world.create_subcomm(split)
        
        #For each column create comm_couple 
        split = dict([(f"comm_couple{n:03d}",row) for n,row in enumerate(self.pe_matrix.T)])
        self.comm_couple = self.comm_world.create_subcomm(split)
        
        #For 1st column create comm_filter
        split = dict([(f"comm_filter{n:03d}",row) for n,row in enumerate(self.pe_matrix[:1,:])])
        self.comm_filter = self.comm_world.create_subcomm(split)
        
        #Now close process that aren't being used. Don't use blocking call to comm_world after this
        if self.comm_world.mype > np.max(self.pe_matrix):
            self.finalize()
            
    def finalize(self):
        """ Shut down the processes."""
        self.comm_world.comm.Barrier()
        self.comm_world.finalize()
        
    def for_comm_model(self, member):
        """
        Indicates whether ensemble member uses comm_model to run forward.

        Parameters
        ----------
        member : int>=0
            Index of ensemble member.
            
        Returns
        -------
        Bool indicating whether member uses comm_model.
        
        """
        #Assign ensemble members in round robin fashion.
        model_id = np.mod(member, self.pe_matrix.shape[0]) 
        return self.comm_model.name == f"comm_model{model_id:03d}"
    
    def get_pe_index(self, pe):
        """
        Get index of comm_world process pe in the pe_matrix. 
        """
        if pe>=np.size(self.pe_matrix):
            return (None, None)
        else:
            return np.unravel_index(pe, self.pe_matrix.shape)
            
    @property 
    def this_comm_model(self):
        """Numerical index of the comm_model communicator."""
        ind = self.get_pe_index(self.comm_world.mype)
        return ind[0]
        
    @property 
    def n_comm_models(self):
        """Total number of comm_model communicators."""
        return np.size(self.pe_matrix, 0)
    
    @property 
    def this_comm_couple(self):
        """Numerical index of the comm_couple communicator."""
        ind = self.get_pe_index(self.comm_world.mype)
        return ind[1]
        
    @property 
    def n_comm_couples(self):
        """Total number of comm_comm communicators."""
        return np.size(self.pe_matrix, 1)
    
    @property 
    def this_comm_filter(self):
        """Numerical index of the comm_filter communicator."""
        ind = self.get_pe_index(self.comm_world.mype)
        return ind[0] if ind[0]==0 else None  
        
    @property 
    def n_comm_filters(self):
        """Total number of comm_filter communicators."""
        return 1
    
    def sum(self, send):
        recv = 0 
        recv = MPI.allreduce(send, recv, MPI.SUM)
        return recv
        
    def _create_pe_matrix(self):
        """
        Create matrix with processes. Each row forms a comm_model,
        each column a comm_couple. 
        """
        #Effective number of processes that can be used. 
        npes = int(self.comm_world.npes / self.npes_per_model)
        npes = npes * self.npes_per_model
        self.pe_matrix = np.arange(npes).reshape((-1, self.npes_per_model))
        
    def throw_error(self, message):
        self.comm_world.throw_error(message)
