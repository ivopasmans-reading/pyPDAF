from models import Ar1Model 
from observations import PointObservation, DictObsReader
from analyzers import AnalyzerBuilder, ETKF
from covariances import DiagonalObsCovariance
from localization import RLocalizer, DistType
from parallelization import PdafParallelization 
from inflation import FixedInflator
from general import *
import numpy as np
import pyPDAF.PDAF as PDAF
from dasystems import DAS

def create_obs_from_truth(truth,indt,ind0,ind1,sigo):
    """ 
    Create observation from artificial truth. 
    
    Parameters
    ----------
    truth : models.Model object 
        Truth model after completion of its model run. 
    indt : list of int 
        Time indices for which observations are created. 
    ind0,iind1 : list of int 
        Observations are made at points (i0,i1) with i0 in 
        ind0 and i1 in ind1 for every combo of i0,i1.
    sigo : float 
        Observational error standard deviation. 
    
    Returns
    -------
    obs : observations.PointObservation 
        Observation operator associated with these observations. 
    """
    ocoords   = np.array([(it,i0,i1) for it in indt
                          for i0 in ind0
                          for i1 in ind1])
    observed  = np.array([truth.saved_output[it,i0,i1] for it,i0,i1 in ocoords])
    observed += sigo*np.random.normal(size=np.shape(observed))
    ocoords = np.array(ocoords, dtype=float)
    reader = DictObsReader(control.comm_filter,
                           {'observed': observed,
                            'coord':ocoords[:,1:],'time':ocoords[:,0]*truth.dt,
                            'variance':sigo**2*np.ones_like(observed)})

    obs = PointObservation(control.comm_filter, covariance, reader, truth.nn_interpolator)
    obs.create_windows_from_model(truth.time_init, truth.save_steps[-1], truth.dt)
    print(reader.data)
    return obs

def calculate_stats(truth, models):
    """ Calculate statistics of ensemble. """
    output = np.array([model.saved_output for model in models])
    ref    = np.array([truth.saved_output for _ in models])
    output = output.reshape(output.shape[:2] + (-1,))
    ref    = ref.reshape(ref.shape[:2]+(-1,))
    print(output.shape)
    
    stats = {}
    stats['spread'] = np.sqrt(np.mean(np.var(output[:,:,:1], axis=0), axis=-1))

    output, ref = np.mean(output,0), np.mean(ref, 0)
    stats['time'] = truth.save_steps * truth.dt
    stats['bias'] = np.mean(output - ref, axis=-1)
    stats['rmse'] = np.sqrt(np.mean((output-ref)**2, axis=-1))
    stats['correlation'] = np.array([np.corrcoef(o,r)[0,1] for o,r in zip(output,ref)])
    return stats

#Create process control. Only single process is used here. 
control = PdafParallelization(1)
control.initialize()
#Create truth 
dt_da = 1
config = {'dt':10., 'shape':[6,20], 'var':1.0, 'corr':.7, 'seed':1000, 'save_steps':range(0,5,dt_da)}
truth = Ar1Model(**config)
truth.init_fields(control.comm_model)
while truth.step < truth.save_steps[-1]:
    truth.step_forward(truth.step, dt_da)
#Create model for DA. 
models_da = [Ar1Model(**{**config,'seed':seed}) for seed in range(2000,2400,20)]
#Create reference without da
models_no = [Ar1Model(**{**config,'seed':seed}) for seed in range(2000,2400,20)]
for model in models_no:
    model.init_fields(control.comm_model)
    while model.step < truth.save_steps[-1]:
        model.step_forward(model.step_init, dt_da)
#Create localizer. By default no localization is used. 
localizer = RLocalizer(pe=control.comm_filter, projection=DistType.cartesian)
#Covariance. By default errors are assumed to Gaussian. 
covariance = DiagonalObsCovariance(pe=control.comm_filter, localizer=localizer)
#No inflation 
inflator = FixedInflator(control.comm_filter)
#Filter. In this case ETKF
builder = AnalyzerBuilder()
analyzer = ETKF(pe=control.comm_filter, covariance=covariance, inflator=inflator)
#Create two types of observations
obs1 = create_obs_from_truth(truth, range(1,truth.save_steps[-1]+1), 
                             range(2,truth.shape[0],6), range(2,truth.shape[1],2), .05)
obs2 = create_obs_from_truth(truth, range(1,truth.save_steps[-1]+1), range(4,truth.shape[0],6), range(2,truth.shape[1],2), .05)
observations = [obs1,obs2]
#Create and run DA system 
das = DAS(control, models_da, observations, analyzer, builder)
das.run(truth.save_steps[-1])
#Calculate statics
stats_no, stats_da = calculate_stats(truth, models_no), calculate_stats(truth, models_da)
print()
print('No/DA RMSE', stats_no['rmse'], stats_da['rmse'])
print('No/DA correlation', stats_no['correlation'], stats_da['correlation'])

control.finalize()

