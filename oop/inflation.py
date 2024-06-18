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
from abc import ABC 
from enum import IntEnum
from dataclasses import dataclass
from parallelization import ProcessControl

class TypeForget(IntEnum):
    """ Type of inflatin schemes. """
    fixed = 0 
    global_adaptive = 1
    local_adaptive = 2

@dataclass 
class Inflator:
    """ 
    Abstract class dealing with covariance inflation. 
    
    Attributes 
    ----------
    forgetting_factor : float 
        Inverse of inflation factor. 1.0 is no inflation. 
    
    """
    pe : ProcessControl
    inflationtype : TypeForget 
    forgetting_factor : float = 1.0
  
    @property 
    def inflation_factor(self):
        return 1./self.forgetting_factor 
    
    @inflation_factor.setter 
    def inflation_factor(self, factor):
        self.forgetting_factor = 1./factor        

class FixedInflator(Inflator):
    """ Inflation is constant. """
    
    def __init__(self, pe, forgetting_factor=1.0):
        super().__init__(pe=pe, forgetting_factor=forgetting_factor, 
                         inflationtype=TypeForget.fixed)

class AdaptiveInflator(Inflator): 
    """ Inflation depends on observational variance. """
    
    def __init__(self, pe, forgetting_factor=1.0, apply_local=False):
        inflationtype = TypeForget.local_adaptive if apply_local else TypeForget.global_adaptive
        super().__init__(pe=pe, forgetting_factor=forgetting_factor,
                         inflationtype=inflationtype)
        self.apply_local = apply_local

