from copy import deepcopy
import numpy as np
import brandywine.conservative_vars as cv
import brandywine.boundary_conds as bc
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
        self.U0 = deepcopy(self.U)
        # TODO string inputs for flux scheme
        # TODO string inputs for time scheme
        # TODO initialize output data object - output to vtk, csv, etc

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

        p_arr = np.zeros(self.x.ntotcells)
        p_arr[left_mask] = p_left
        p_arr[right_mask] = p_right

        r_arr = np.zeros(self.x.ntotcells)
        r_arr[left_mask] = rho_left
        r_arr[right_mask] = rho_right

        u_arr = np.zeros(self.x.ntotcells)

        return cv.ConservativeVars(
            r_arr, r_arr*u_arr, total_internal_energy(p_arr, r_arr, u_arr,
                                                      self.gam)
        )
    
    def solve(self):
        for i in range(self.times.size):
            self.times[i] = self.minimum_timestep()
            self.update_bc()
            # flux schemes plugin here
            self.U0 = self.U

    def minimum_timestep(self):
        return self.cfl*self.x.dx.min() / np.max(self.U.u + self.a)
    
    def update_bc(self):
        self.U[0] = bc.inviscid_wall0(self.U[1])
        self.U[-1] = bc.inviscid_wall0(self.U[-2])

    @property
    def p(self):
        return pressure(self.U.e, self.U.ke, self.gam)
    
    @property
    def a(self):
        return sound_speed(self.p, self.U.r, self.gam)
    
def pressure(total_internal_e, kinetic_e, gam:float):
    return (gam-1) * (total_internal_e - kinetic_e)
    
def sound_speed(p, rho, gam:float):
    return np.sqrt(gam*p/rho)
    
def total_internal_energy(p, rho, u, gam:float):
    return p/(gam-1) + 0.5*rho*u*u