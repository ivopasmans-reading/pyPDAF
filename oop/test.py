from models import Ar1Model 
from observations import PointObservation, DictObsReader
from analyzers import EnsembleGlobalBuilder
from covariances import DiagonalObsCovariance
from localization import RLocalizer
from parallelization import PdafParallelization 
from options import *
import numpy as np
import pyPDAF.PDAF as PDAF


#Create process control. Only single process is used here. 
control = PdafParallelization(1)
control.initialize()
#Create truth 
dt, final_step = 10., 100
model = Ar1Model(dt,[6,20], 1.0, .8, 1000)
model.init_fields()
truth = [model.values+0.0]
while model.step < final_step:
    model.step_forward(model.step, 10)
    truth.append(model.values + 0.0)
#Create model for DA. 
model = Ar1Model(dt, [6,20], 1.0, .8, 2000)
#Create localizer. By default no localization is used. 
localizer = RLocalizer(DistType.cartesian)
#Covariance. By default errors are assumed to Gaussian. 
covariance = DiagonalObsCovariance(localizer)
#Filter. In this case ETKF
builder = EnsembleGlobalBuilder(FilterType.etkf,0,[],[])
#Observations 
ocoords = np.array([(i0,i1,it) for i0 in [2.,4.] for i1 in [5.,15.] for it in range(10,final_step,10)])
reader = DictObsReader({'value':np.zeros_like(ocoords)[:,0],'coord':ocoords[:,:2],'time':ocoords[:,-1],
                        'variance':.1*np.ones_like(ocoords)[:,0]})
obs = PointObservation(control.comm_filter, covariance, reader, model.nn_interpolator)
PDAF.omi_init(1)
obs.create_windows_from_model(0,final_step,dt)
obs.init(1,10.)
truth0 = np.asfortranarray(np.reshape(truth[0],(-1,)))
print(len(truth0))
ostate=obs.obs_op(truth0)
control.finalize()

