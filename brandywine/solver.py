from copy import deepcopy
import numpy as np
import brandywine.conservative_vars as cv
import brandywine.boundary_conds as bc
import brandywine.state_equations as eos
import brandywine.time_schemes.temporal_flux as tfx
import brandywine.spatial_schemes.inviscid_flux as ifx
import brandywine.grid as grd

class ShockTubeSolver:
    def __init__(self, p_left:float, rho_left:float,
                 p_right:float, rho_right:float,
                 ncells:int, ntimesteps:int, cfl:float,
                 L:float=None, L_left:float=None, L_right:float=None,
                 gam:float=1.4, Rgas:float=287.,
                 inviscid_scheme:str="lax", time_scheme:str="rk1"):
        self.gam = gam
        self.Rgas = Rgas
        self.timesteps = np.zeros(ntimesteps)
        self.cfl = cfl
        self.x = self._init_grid(ncells, L, L_left, L_right)
        self.U = self._init_cvars(p_left, rho_left, p_right, rho_right)
        self.U0 = deepcopy(self.U)
        self.spatial_scheme = ifx.get_inviscid_scheme(inviscid_scheme)
        self.time_scheme = tfx.get_time_scheme(time_scheme)
        # TODO initialize output data object - output to vtk, csv, etc

    @property
    def total_time(self):
        return np.sum(self.timesteps)

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

        pressure = np.zeros(self.x.ntotcells)
        pressure[left_mask] = p_left
        pressure[right_mask] = p_right

        density = np.zeros(self.x.ntotcells)
        density[left_mask] = rho_left
        density[right_mask] = rho_right

        velocity = np.zeros(self.x.ntotcells)

        return cv.ConservativeVars(
            density=density,
            momentum=density*velocity,
            total_energy=eos.total_energy(pressure, 0.5*density*velocity**2, self.gam)
        )
    
    def solve(self):
        for i in range(self.timesteps.size):
            self.timesteps[i] = self.minimum_timestep()
            self.update_bc()
            for j in self.x.range_cells:
                self.U[j] = self.time_scheme(
                    U0 = self.U0[j],
                    dt = self.timesteps[i],
                    spatial_derivative = ifx.spatial_derivative(
                        U = self.U0,
                        index = j,
                        gam = self.gam,
                        dx = self.x[j+1]-self.x[j],
                        dt = self.timesteps[i],
                        flux_func = self.spatial_scheme
                    )
                )
            self.update_previous_soln()

    def minimum_timestep(self):
        return self.cfl*self.x.dx.min() / np.max(self.U.velocity + self.a())
    
    def update_bc(self):
        self.U[0] = bc.inviscid_wall0(self.U[1])
        self.U[-1] = bc.inviscid_wall0(self.U[-2])

    def update_previous_soln(self):
        self.U0[:] = self.U[:]
    
    def a(self):
        return eos.sound_speed(
            pressure=eos.pressure(self.U.total_energy, self.U.kinetic_energy, self.gam),
            density=self.U.density,
            gam=self.gam
        )