import numpy as np
import pytest
import brandywine.solver as sol

def test_pressure():
    assert sol.pressure(259437.5, 0.5*1.225*100**2, 1.4) == pytest.approx(101325)

def test_sound_speed():
    assert sol.sound_speed(1e5, 1.2, 1.3) == pytest.approx(329.14029430219165)

def test_total_internal_energy():
    p=101325
    rho=1.225
    u = 100
    assert sol.total_internal_energy(p, rho, u, gam=1.4) == pytest.approx(259437.5)
    assert sol.total_internal_energy(100, 1.2, 5000, 1.3) == pytest.approx(15000333.333333334)

class TestShockTubeSolver:
    p_left = 1e5
    p_right = 1e5
    rho_left = 4
    rho_right = 4
    gam = 1.3
    Rgas = 289
    cfl = 2.1
    ntimesteps = 200
    ncells = 100
    sod = sol.ShockTubeSolver(p_left, rho_left, p_right, rho_right,
                              ncells, L=3., gam=gam, Rgas=Rgas, cfl=cfl,
                              ntimesteps=ntimesteps)
    
    def test_cfl(self):
        assert self.sod.cfl == self.cfl

    def test_times(self):
        np.testing.assert_equal(self.sod.times, np.zeros(self.ntimesteps))
    
    def test_gam(self):
        assert self.sod.gam == self.gam

    def test_Rgas(self):
        assert self.sod.Rgas == self.Rgas
    
    def test_p(self):
        np.testing.assert_allclose(self.sod.p, np.zeros(self.ncells+2)+1e5)

    def test_minimum_timestep(self):
        assert self.sod.minimum_timestep() == pytest.approx(0.00034946112362189437)

    def test_a(self):
        np.testing.assert_allclose(
            self.sod.a,
            np.zeros(self.ncells+2) + np.sqrt(1.3*1e5/4)
        )