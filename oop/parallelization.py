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
from abc import ABC, abstractmethod 
import sys

class ProcessControl:
    """
    Handles parallelization to be handled in a generic way in the code. 
    
    This class contains several methods that the codes needs to 
    run in parallel but in such a way that they are available even
    if the parallellization scheme doesn't supply them. By default
    serial run is assumed. 
    """
    
    def __init__(self, name="world"):
        self.name = name
        self.comm = None
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
    
class MpiControl(ProcessControl):
    """ Communicator using created using mpi4py. """
    
    def __init__(self, comm=None, name="world"):
        """ Constructor. Default comm is mpi_comm_world."""
        self.name = name 
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
            new  = MpiControl(comm, "sub_"+self.name+"_undefined")
        elif len(keys)==1:
            color, name = keys[0]
            comm = self.comm.Split(color, self.mype)
            new  = MpiControl(comm, name)
        else:
            raise ValueError("Process can only be part of a single subcommunicator.")
        
        new.parent = self 
        return new
        
    def py2f(self):
        return self.comm.py2f()
        
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
            return self.unravel_index(pe, self.ge_matrix.shape)
            
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
        
    def _create_pe_matrix(self):
        """
        Create matrix with processes. Each row forms a comm_model,
        each column a comm_couple. 
        """
        #Effective number of processes that can be used. 
        npes = int(self.comm_world.npes / self.npes_per_model)
        npes = npes * self.npes_per_model
        self.pe_matrix = np.arange(npes).reshape((-1, self.npes_per_model))
        
class parallelization:

    """Summary

    Attributes
    ----------
    COMM_couple : `MPI.Comm`
        model and filter coupling communicator
    COMM_filter : `MPI.Comm`
        filter communicator
    COMM_model : `MPI.Comm`
        model communicator
    filterpe : bool
        whether the PE is used for filter
    local_npes_model : int
        number of model PEs each member
    mype_filter : int
        rank of fileter communicator
    mype_model : int
        rank of model communicator
    mype_world : int
        rank of world communicator
    n_filterpes : int
        number of filter PEs
    n_modeltasks : int
        number of model tasks
    npes_filter : int
        number of filter PEs
    npes_model : int
        number of model PEs
    npes_world : int
        number of global PEs
    task_id : int
        ensemble id
    """

    def __init__(self, dim_ens, n_modeltasks, screen):
        """Init the parallization required by PDAF

        Parameters
        ----------
        dim_ens : TYPE
            Description
        n_modeltasks : int
            Number of model tasks/ ensemble size
            This parameter should be the same
            as the number of PEs
        screen : int
            The verbose level of screen output.
            screen = 3 is the most verbose level.
        """

        self.n_modeltasks = n_modeltasks
        self.n_filterpes = 1

        self.init_parallel()
        self.getEnsembleSize()

        # Initialize communicators for ensemble evaluations
        if (self.mype_world == 0):
            print(('Initialize communicators for assimilation with PDAF'))

        self.isCPUConsistent()
        self.isTaskConsistent(n_modeltasks)

        # get ensemble communicator
        self.getModelComm()
        self.getModelPERankSize()

        if (screen > 1):
            print(('MODEL: mype(w)= ', self.mype_world,
                   '; model task: ', self.task_id,
                   '; mype(m)= ', self.mype_model,
                   '; npes(m)= ', self.npes_model))

        # Generate communicator for filter
        self.getFilterComm()
        self.getFilterPERankSize()

        # Generate communicators for communication
        self.getCoupleComm()

        self.printInfo(screen)

    def printInfo(self, screen):
        """print parallelization info

        Parameters
        ----------
        screen : int
            The verbose level of screen output.
        """
        # *** local variables ***
        #  Rank and size in COMM_couple
        mype_couple = self.COMM_couple.Get_rank()
        #  Variables for communicator-splitting
        color_couple = self.mype_model + 1

        if (screen > 0):
            if (self.mype_world == 0):
                print(('PE configuration:'))
                print(('world', 'filter', '   model    ',
                       '   couple   ', 'filterPE'))
                print(('rank ', ' rank ', 'task',
                       'rank', 'task', 'rank', 'T/F'))
                print(('--------------------------------------'))
            MPI.COMM_WORLD.Barrier()
            if (self.task_id == 1):
                print((self.mype_world, self.mype_filter, self.task_id,
                       self.mype_model, color_couple,
                       mype_couple, self.filterpe))
            MPI.COMM_WORLD.Barrier()
            if (self.task_id > 1):
                print((self.mype_world, ' ', self.task_id, self.mype_model,
                       color_couple, mype_couple, self.filterpe))
            MPI.COMM_WORLD.Barrier()
            if (self.mype_world == 0):
                print('')

    def init_parallel(self):
        """Initialize MPI

        Routine to initialize MPI, the number of PEs
        (npes_world) and the rank of a PE (mype_world).
        The model is executed within the scope of the
        communicator Comm_model. It is also initialized
        here together with its size (npes_model) and
        the rank of a PE (mype_model) within Comm_model.
        """

        if not MPI.Is_initialized():
            MPI.Init()

        # Initialize model communicator, its size and the process rank
        # Here the same as for MPI_COMM_WORLD
        self.COMM_model = MPI.COMM_WORLD
        self.npes_model = None
        self.npes_world = None
        self.mype_model = None
        self.mype_world = None

        self.npes_world = self.COMM_model.Get_size()
        self.mype_world = self.COMM_model.Get_rank()

    def getModelComm(self):
        """get model communicator
        Generate communicators for model runs
        (Split COMM_ENSEMBLE)
        """
        self.getPEperModel()

        pe_index = np.cumsum(self.local_npes_model, dtype=int)

        mype_ens = self.mype_world

        if mype_ens + 1 <= self.local_npes_model[0]:
            self.task_id = 1
        else:
            self.task_id = np.where(pe_index < mype_ens + 1)[0][-1] + 2

        self.COMM_model = MPI.COMM_WORLD.Split(self.task_id, mype_ens)

    def getFilterComm(self):
        """Generate communicator for filter
        """
        # Init flag FILTERPE (all PEs of model task 1)
        self.filterpe = True if self.task_id == 1 else False

        my_color = self.task_id if self.filterpe \
            else MPI.UNDEFINED

        self.COMM_filter = MPI.COMM_WORLD.Split(my_color,
                                                self.mype_world)

    def getCoupleComm(self):
        """Generate communicator for filter
        """
        # Init flag FILTERPE (all PEs of model task 1)
        color_couple = self.mype_model + 1

        self.COMM_couple = MPI.COMM_WORLD.Split(color_couple,
                                                self.mype_world)

    def getModelPERankSize(self):
        """get model PE rank and size
        """
        self.npes_model = self.COMM_model.Get_size()
        self.mype_model = self.COMM_model.Get_rank()

    def getFilterPERankSize(self):
        """get filter PE rank and size
        """
        self.npes_filter = self.COMM_model.Get_size()
        self.mype_filter = self.COMM_model.Get_rank()

    def getEnsembleSize(self):
        """Parse number of model tasks

        The module variable is N_MODELTASKS.
        Since it has to be equal to the ensemble size
        we parse dim_ens from the command line.
        """
        # handle for command line parser
        # handle = 'dim_ens'
        # parse(handle, self.n_modeltasks)
        pass

    def getPEperModel(self):
        """Store # PEs per ensemble
        used for info on PE 0 and for generation
        of model communicators on other Pes
        """

        self.local_npes_model = np.zeros(self.n_modeltasks, dtype=int)

        self.local_npes_model[:] = np.floor(
            self.npes_world/self.n_modeltasks)

        size = self.npes_world \
            - self.n_modeltasks * self.local_npes_model[0]
        self.local_npes_model[:size] = self.local_npes_model[:size] + 1

    def isCPUConsistent(self):
        """Check consistency of number of parallel ensemble tasks
        """
        pass
        if self.n_modeltasks > self.npes_world:
            # number of parallel tasks is set larger than available PEs ***
            self.n_modeltasks = self.npes_world
            if self.mype_world == 0:
                print('!!! Resetting number of parallel ensemble'
                      ' tasks to total number of PEs!')

    def isTaskConsistent(self, dim_ens):
        """Check consistency of number of model tasks

        Parameters
        ----------
        dim_ens : int
            ensemble size
        """

        # For dim_ens=0 no consistency check
        # for the ensemble size with the
        # number of model tasks is performed.
        if (dim_ens <= 0):
            return

        # Check consistency with ensemble size
        if (self.n_modeltasks > dim_ens):
            # parallel ensemble tasks is set larger than ensemble size
            self.n_modeltasks = dim_ens

            if (self.mype_world == 0):
                print(('!!! Resetting number of parallel'
                       'ensemble tasks to number of ensemble states!'))

    def finalize_parallel(self):
        """Finalize MPI
        """
        MPI.COMM_WORLD.Barrier()
        MPI.Finalize()

    def abort_parallel(self):
        """Abort MPI
        """
        MPI.COMM_WORLD.Abort(1)
