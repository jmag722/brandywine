import pytest
import brandywine.time_schemes.temporal_flux as tfx
import brandywine.time_schemes.runge_kutta as rk

def test_get_time_scheme_rk1():
    with pytest.raises(KeyError):
        tfx.get_time_scheme("hey")
    assert tfx.get_time_scheme("RK1") == rk.runge_kutta1
    assert tfx.get_time_scheme("Runge-kutta1") == rk.runge_kutta1
    assert tfx.get_time_scheme("Forward-Euler") == rk.runge_kutta1
    assert tfx.get_time_scheme("FE") == rk.runge_kutta1