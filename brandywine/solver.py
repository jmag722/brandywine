import numpy as np
import brandywine.conservative_vars as cv
import brandywine.grid as grd

class ShockTubeSolver:
    def __init__(self, p_left:float, rho_left:float,
                 p_right:float, rho_right:float,
                 ncells:int, ntimesteps:int, cfl:float,
                 L:float=None, L_left:float=None, L_right:float=None,
                 gam:float=1.4, Rgas:float=287.):
        self.gam = gam
        self.Rgas = Rgas
        self.times = np.zeros(ntimesteps)
        self.cfl = cfl
        self.x = self._init_grid(ncells, L, L_left, L_right)
        self.U = self._init_cvars(p_left, rho_left, p_right, rho_right)

    def _init_grid(self, ncells:int, L=None, L_left=None, L_right=None):
        if L is None and all([v is not None for v in [L_left, L_right]]):
            _L_left = L_left
            _L_right = L_right
        else:
            _L_left = 0.5*L
            _L_right = 0.5*L
        x = np.linspace(-1*_L_left, _L_right, ncells+1, endpoint=True, dtype=np.float64)
        return grd.Grid1D(x)

    def _init_cvars(self, p_left, rho_left, p_right, rho_right):
        left_mask = self.x.cc <= 0.
        right_mask = self.x.cc > 0.
        p_arr = np.zeros(self.x.ncells)
        p_arr[left_mask] = p_left
        p_arr[right_mask] = p_right
        r_arr = np.zeros(self.x.ncells)
        r_arr[left_mask] = rho_left
        r_arr[right_mask] = rho_right
        u_arr = np.zeros(self.x.ncells)
        return cv.ConservativeVars(r_arr, r_arr*u_arr, etot(p_arr, r_arr, u_arr, self.gam))

    @property
    def p(self):
        return (self.gam-1) * (self.U.e + self.U.ke)

    @property
    def a(self):
        return np.sqrt(self.gam*self.p/self.U.r)
    
def etot(p, rho, u, gam:float):
    return p/(gam-1) + 0.5*rho*u*u