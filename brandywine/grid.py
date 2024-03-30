import numpy as np

class Grid1D:
    nghosts:int = 2
    def __init__(self, x:np.ndarray):
        self._x = np.zeros(x.size+Grid1D.nghosts)
        self._x[self.istart_vert: self.iend_vert+1] = x
        self._x[0] = 2*x[0] - x[1]
        self._x[-1] = 2*x[-1] - x[-2]

    @property
    def istart_cell(self):
        return Grid1D.nghosts//2
    @property
    def istart_vert(self):
        return Grid1D.nghosts//2
    
    @property
    def iend_cell(self):
        return self.istart_cell + self.ncells - 1
    @property
    def iend_vert(self):
        return self.istart_vert + self.nvertices - 1

    @property
    def range_cells(self):
        return range(self.istart_cell, self.iend_cell + 1)
    
    @property
    def ntotvertices(self):
        return self._x.size
        
    @property
    def ntotcells(self):
        return self.ntotvertices - 1
    
    @property
    def nvertices(self):
        return self.ntotvertices - Grid1D.nghosts
        
    @property
    def ncells(self):
        return self.ntotcells - Grid1D.nghosts
    
    @property
    def dx(self):
        return np.diff(self._x)
    
    @property
    def cc(self):
        return self._x[:-1] + self.dx*0.5

    def __getitem__(self, key:int):
        return self._x[key]
    
    # def __setitem__(self, key:int, value:float):
    #     self._x[key] = value

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