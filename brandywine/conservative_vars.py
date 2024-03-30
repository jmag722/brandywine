from enum import IntEnum, unique
import numpy as np

@unique
class Index(IntEnum):
    RHO = 0
    RHOU = 1
    E = 2
    SIZE = 3

class ConservativeVars:
    def __init__(self, rho:np.ndarray, rhou:np.ndarray, e:np.ndarray):
        assert rho.shape == rhou.shape
        assert rhou.shape == e.shape
        ncells = rho.size
        self._arr = np.empty((ncells, Index.SIZE), dtype=np.float64, order="C")
        self.r = rho
        self.ru = rhou
        self.e = e

    @property
    def r(self):
        return self._arr[:, Index.RHO]
    
    @r.setter
    def r(self, density):
        self._arr[:, Index.RHO] = density
    
    @property
    def ru(self):
        return self._arr[:, Index.RHOU]

    @ru.setter
    def ru(self, momentum):
        self._arr[:, Index.RHOU] = momentum

    @property
    def ru2(self):
        return self.ru * self.ru / self.r
    
    @property
    def ke(self):
        return 0.5 * self.ru2
    
    @property
    def u(self):
        return self.ru / self.r
    
    @property
    def e(self):
        return self._arr[:, Index.E]

    @e.setter
    def e(self, energy):
        self._arr[:, Index.E] = energy

    def __getitem__(self, key:int):
        return self._arr[key]
    
    def __setitem__(self, key:int, value:np.ndarray):
        self._arr[key] = value

    def __array__(self):
        return self._arr
    
    @property
    def size(self):
        return self._arr.shape[0]
    
    @property
    def shape(self):
        return self._arr.shape