from enum import IntEnum

#Nan value for integers. 
iNAN = -999
#Nan value for floats. 
fNAN = -9.99e9
#Small value 
fEPS = 1.0e-8

def check_init(func):
    def wrapper(self, *args, **kwargs):
        is_init = hasattr(self, 'is_initialized') and self.is_initialized 
        object_str = str(self.__str__)
        func_str = str(func.__str__)
        msg = f"Object {object_str} needs to be initialized before {func_str} can be used."
        if not is_init and hasattr(self, 'pe'):
            self.pe.throw_error(msg)
        elif not is_init:
            raise RuntimeError(msg)
        else:
            return func(self, *args, **kwargs)
    return wrapper

class ScreenOutput(IntEnum):
    """
    This class contains options to print PDAF 
    output to screen.  
    """
    quiet = 0 
    standard = 1
    timing = 2
    
class PdafError(IntEnum):
    """ 
    This class contains possible ways PDAF can fail.
    """
    none = 0
    
    

            

