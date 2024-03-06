import numpy as np
import brandywine.boundary_conds as bc

def test_inviscid_wall0():
    uvec = np.array([1.2, 1200.1, 1e4])
    np.testing.assert_equal(bc.inviscid_wall0(uvec), np.array([1.2, -1200.1, 1e4]))