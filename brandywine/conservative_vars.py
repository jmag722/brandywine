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
        return self._arr[:, Index.RHO]
    
    @property
    def velocity(self):
        return self.momentum / self.density
    
    @property
    def kinetic_energy(self):
        return 0.5 * self.momentum**2 / self.density
    
    @property
    def momentum(self):
        return self._arr[:, Index.MOMX]

    @property
    def total_energy(self):
        return self._arr[:, Index.TOTENERGY]

    def __getitem__(self, key):
        return self._arr[key]
    
    def __setitem__(self, key, value:np.ndarray):
        self._arr[key] = value

    def __array__(self):
        return self._arr
    
def density(cvar:np.ndarray):
    return cvar[Index.RHO]

def momentum(cvar:np.ndarray):
    return cvar[Index.MOMX]

def velocity(cvar:np.ndarray):
    return cvar[Index.MOMX] / cvar[Index.RHO]

def kinetic_energy(cvar:np.ndarray):
    return 0.5 * cvar[Index.MOMX] * cvar[Index.MOMX] / cvar[Index.RHO]

def total_energy(cvar:np.ndarray):
    return cvar[Index.TOTENERGY]