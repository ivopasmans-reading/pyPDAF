from parallelization import parallelization
import Model
import OBS
import DAS


from model import shift
import numpy as np
from mpi4py import MPI

USE_PDAF = True
nt = 100
# obs = np.load('lib/trajectory.npz')['obs']

if USE_PDAF:
    pe = parallelization(dim_ens=9, n_modeltasks=9, screen=0)

# Initial Screen output
if (pe.mype_world==0):
    print('+++++ PDAF online mode +++++')
    print('2D model without parallelization')


# Initialize model
model = Model.Model((18, 36), nt=18, pe=pe)
obs = []
for typename in ['A', 'B']:
    obs.append(OBS.OBS(typename, pe.mype_filter, 
                        model.nx, 1, 2, 0.5))

das = DAS.DAS(pe, model, obs)
das.init()

for it in range(2):
    das.forward(it)

pe.finalize_parallel()