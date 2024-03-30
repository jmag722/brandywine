from enum import IntEnum, unique
import numpy as np

@unique
class Index(IntEnum):
    RHO = 0
    MOMX = 1
    TOTENERGY = 2
    SIZE = 3

class ConservativeVars:
    def __init__(self, density:np.ndarray, momentum:np.ndarray, total_energy:np.ndarray):
        assert density.shape == momentum.shape
        assert momentum.shape == total_energy.shape
        ncells = density.size
        self._arr = np.empty((ncells, Index.SIZE), dtype=np.float64)
        self._arr[:, Index.RHO] = density
        self._arr[:, Index.MOMX] = momentum
        self._arr[:, Index.TOTENERGY] = total_energy

    @property
    def density(self):
        return density(self._arr)
    
    @property
    def momentum(self):
        return momentum(self._arr)

    @property
    def total_energy(self):
        return total_energy(self._arr)

    def __getitem__(self, key):
        return self._arr[key]
    
    def __setitem__(self, key, value:np.ndarray):
        self._arr[key] = value

    def __array__(self):
        return self._arr
    
    @property
    def size(self):
        return self._arr.shape[0]
    
    @property
    def shape(self):
        return self._arr.shape
    
def density(cvar:np.ndarray):
    return cvar[..., Index.RHO]

def momentum(cvar:np.ndarray):
    return cvar[..., Index.MOMX]

def velocity(cvar:np.ndarray):
    return momentum(cvar) / density(cvar)

def kinetic_energy(cvar:np.ndarray):
    return 0.5 * momentum(cvar) * velocity(cvar)

def total_energy(cvar:np.ndarray):
    return cvar[..., Index.TOTENERGY]