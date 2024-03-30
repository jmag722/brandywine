import numpy as np
import pytest
import brandywine.state_equations as eos

def test_pressure():
    assert eos.pressure(259437.5, 0.5*1.225*100**2, 1.4) == pytest.approx(101325)

def test_sound_speed():
    assert eos.sound_speed(1e5, 1.2, 1.3) == pytest.approx(329.14029430219165)

def test_total_energy():
    p=101325
    rho=1.225
    u = 100
    assert eos.total_energy(p, 0.5*rho*u**2, gam=1.4) == pytest.approx(259437.5)
    assert eos.total_energy(100, 0.5*1.2*5000**2, 1.3) == pytest.approx(15000333.333333334)