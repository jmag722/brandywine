import numpy as np

class Grid1D:
    def __init__(self, x:np.ndarray):
        self._x = x
    
    @property
    def nvertices(self):
        return self._x.size
        
    @property
    def ncells(self):
        return self.nvertices - 1
    
    @property
    def dx(self):
        return np.diff(self._x)
    
    @property
    def cc(self):
        return self._x[:-1] + self.dx*0.5

    def __getitem__(self, key:int):
        return self._x[key]
    
    def __setitem__(self, key:int, value:float):
        self._x[key] = value

    def __lt__(self, other):
        return self._x < other
    
    def __gt__(self, other):
        return self._x > other
    
    def __ge__(self, other):
        return self._x >= other
    
    def __le__(self, other):
        return self._x <= other

    def __array__(self):
        return self._x