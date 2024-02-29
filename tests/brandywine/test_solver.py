import numpy as np
import pytest
import brandywine.solver as sol

def test_etot():
    p=101325
    rho=1.225
    u = 100
    assert sol.etot(p, rho, u, gam=1.4) == pytest.approx(259437.5)
    assert sol.etot(100, 1.2, 5000, 1.3) == pytest.approx(15000333.333333334)

class TestShockTubeSolver:
    p_left = 1e5
    p_right = 1e5
    rho_left = 4
    rho_right = 4
    gam = 1.3
    Rgas = 289
    sod = sol.ShockTubeSolver(p_left, rho_left, p_right, rho_right,
                              100, L=3., gam=gam, Rgas=Rgas)
    
    def test_gam(self):
        assert self.sod.gam == self.gam

    def test_Rgas(self):
        assert self.sod.Rgas == self.Rgas
    
    def test_p(self):
        np.testing.assert_allclose(self.sod.p, np.zeros(100)+1e5)

    def test_a(self):
        np.testing.assert_allclose(
            self.sod.a,
            np.zeros(100) + np.sqrt(1.3*1e5/4)
        )