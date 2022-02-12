"""Implementation file for the user-defined routines in PDAF
"""
import numpy as np


def py__init_ens_pdaf(filtertype, state_p, uinv, ens_p):
    """default user-defined init_ens_pdaf python function
    
    Parameters
    ----------
    filtertype : int
        Type of filter
    state_p : ndarray
        1D state vector on local PE
    uinv : ndarray
        2D left eigenvector with shape (dim_ens - 1,dim_ens - 1)
    ens_p : TYPE
        Description
    
    Raises
    ------
    RuntimeError
        Description
    """
    raise RuntimeError('...Wrong init_ens_pdaf is called!!!...')

def py__distribute_state_pdaf(state_p):
    """Summary
    
    Parameters
    ----------
    state_p : TYPE
        Description
    
    Raises
    ------
    RuntimeError
        Description
    """
    raise RuntimeError('...Wrong distribute_state_pdaf is called!!!...')

def py__collect_state_pdaf(state_p):
    """Summary
    
    Parameters
    ----------
    state_p : TYPE
        Description
    
    Raises
    ------
    RuntimeError
        Description
    """
    raise RuntimeError('...Wrong collect_state_pdaf is called!!!...')

def py__next_observation_pdaf(stepnow, nsteps, doexit, time):
    """Summary
    
    Parameters
    ----------
    stepnow : TYPE
        Description
    nsteps : TYPE
        Description
    doexit : TYPE
        Description
    time : TYPE
        Description
    
    Raises
    ------
    RuntimeError
        Description
    """
    raise RuntimeError('...Wrong next_observation_pdaf is called!!!...')

def py__prepoststep_ens_pdaf(step, state_p, uinv, ens_p):
    """Summary
    
    Parameters
    ----------
    step : TYPE
        Description
    state_p : TYPE
        Description
    uinv : TYPE
        Description
    ens_p : TYPE
        Description
    
    Raises
    ------
    RuntimeError
        Description
    """
    raise RuntimeError('...Wrong prepoststep_ens_pdaf is called!!!...')


cdef void c__init_ens_pdaf(int* filtertype, int* dim_p, int* dim_ens, 
                           double* state_p, double* uinv, 
                           double* ens_p, int* flag):
    """initialise the state ensemble
    """
    state_p_numpy = np.asarray(<double[:dim_p[0]]> state_p)
    uinv_numpy = np.asarray(
                    <double[:dim_ens[0]-1, :dim_ens[0]-1]> uinv).T
    if (dim_ens[0] != dim_p[0]):
        ens_p_numpy = np.asarray(
                        <double[:dim_ens[0], :dim_p[0]]> ens_p)
    else:
        ens_p_numpy = np.asarray(
                        <double[:dim_ens[0], :dim_ens[0]]> ens_p).T

    flag[0] = py__init_ens_pdaf(filtertype[0], state_p_numpy, 
                                uinv_numpy, ens_p_numpy, flag[0])

cdef void c__distribute_state_pdaf(int* dim_p, double* state_p):
    """distribute state_p to model variables
    """
    state_p_numpy = np.asarray(<double[:dim_p[0]]> state_p)
    py__distribute_state_pdaf(state_p_numpy)

cdef void c__collect_state_pdaf(int* dim_p, double* state_p):
    state_p_numpy = np.asarray(<double[:dim_p[0]]> state_p)
    py__collect_state_pdaf(state_p_numpy)

cdef void c__next_observation_pdaf(int* stepnow, int* nsteps, 
                                    int* doexit, double* time):
    """Setup the timesteps between two analysis
    """
    nsteps[0], doexit[0], time[0] = py__next_observation_pdaf(
                        stepnow[0], nsteps[0], doexit[0], time[0])

cdef void c__prepoststep_ens_pdaf(int* step, int* dim_p, int* dim_ens,
            int* dim_ens_p, int* dim_obs_p, 
            double* state_p, double* uinv, double* ens_p, int* flag):
    """distribute state_p to model variables"""
    
    state_p_numpy = np.asarray(<double[:dim_p[0]]> state_p)
    uinv_numpy = np.asarray(
                    <double[:dim_ens[0]-1, :dim_ens[0]-1]> uinv).T
    if (dim_ens[0] != dim_p[0]):
        ens_p_numpy = np.asarray(
                        <double[:dim_ens[0], :dim_p[0]]> ens_p)
    else:
        ens_p_numpy = np.asarray(
                        <double[:dim_ens[0], :dim_ens[0]]> ens_p).T
    py__prepoststep_ens_pdaf(step[0], state_p_numpy, 
                             uinv_numpy, ens_p_numpy)