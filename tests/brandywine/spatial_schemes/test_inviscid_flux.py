import pytest
import brandywine.spatial_schemes.inviscid_flux as ifx
import brandywine.spatial_schemes.lax_friedrich as lax

def test_get_inviscid_scheme_lax():
    with pytest.raises(KeyError):
        ifx.get_inviscid_scheme("hey")
    assert ifx.get_inviscid_scheme("Lax") == lax.flux
    assert ifx.get_inviscid_scheme("Lax-Friedrich") == lax.flux
    assert ifx.get_inviscid_scheme("LF") == lax.flux