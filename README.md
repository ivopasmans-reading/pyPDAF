# pyPDAF
A (incomplete) Python interface to the Fortran-written data assimilation library - PDAF

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/yumengch/pyPDAF/test_build)

## Prerequisite:
- `PDAF-V1.16`
- `Fortran compiler: e.g.:gfortran/intel fortran`
- `a message passing interface (MPI) implementation: e.g. openMPI/MPICH`
- `Python>=3.8`


## Installation:
- Currently, Fortran-written PDAF is compiled together with pyPDAF. Hence, the Fortran compiler options including needs to be specified in [`setup.py`](setup.py).
- Install Python package: ```pip install -e .```

## Run example:
```mpiexec -n 8 python -u example/main.py```

## Note:
Currently, it only interfaces with limited subroutines of ```PDAF-V1.16``` with an example for online coupling with PDAF using a simple model based on the [tutorial](http://pdaf.awi.de/trac/wiki/FirstSteps) from PDAF. More subroutines will be supported in future release.

## Contributors:
Yumeng Chen, Lars Nerger

pyPDAF is mainly developed and maintainde by National Centre for Earth Observation and University of Reading.

<img src="https://github.com/nansencenter/DAPPER/blob/master/docs/imgs/UoR-logo.png?raw=true" height="140" />
<img src="https://github.com/nansencenter/DAPPER/blob/master/docs/imgs/nceologo1000.png?raw=true" width="400">
