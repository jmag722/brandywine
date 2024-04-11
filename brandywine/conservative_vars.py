from enum import IntEnum, unique
import numpy as np

import brandywine.state_equations as eos

@unique
class Index(IntEnum):
    RHO = 0
    RHOU = 1
    RHOE = 2
    SIZE = 3

class ConservativeVars:
    def __init__(self, density:np.ndarray, momentum:np.ndarray, total_energy:np.ndarray):
        assert density.shape == momentum.shape
        assert momentum.shape == total_energy.shape
        ncells = density.size
        self._arr = np.empty((ncells, Index.SIZE), dtype=np.float64)
        self._arr[:, Index.RHO] = density
        self._arr[:, Index.RHOU] = momentum
        self._arr[:, Index.RHOE] = total_energy

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
        return self._arr[:, Index.RHOU]

    @property
    def total_energy(self):
        return self._arr[:, Index.RHOE]

    def __getitem__(self, key):
        return self._arr[key]
    
    def __setitem__(self, key, value:np.ndarray):
        self._arr[key] = value

    def __array__(self):
        return self._arr
    
def density(cvar:np.ndarray):
    return cvar[Index.RHO]

def momentum(cvar:np.ndarray):
    return cvar[Index.RHOU]

def velocity(cvar:np.ndarray):
    return cvar[Index.RHOU] / cvar[Index.RHO]

def kinetic_energy(cvar:np.ndarray):
    return 0.5 * cvar[Index.RHOU] * cvar[Index.RHOU] / cvar[Index.RHO]

def total_energy(cvar:np.ndarray):
    return cvar[Index.RHOE]

def pressure(cvar:np.ndarray, gam:float):
    return eos.pressure(total_energy=total_energy(cvar),
                        kinetic_energy=kinetic_energy(cvar),
                        gam=gam)

def sound_speed(cvar:np.ndarray, gam:float):
    return eos.sound_speed(pressure=pressure(cvar=cvar, gam=gam),
                           density=density(cvar), gam=gam)