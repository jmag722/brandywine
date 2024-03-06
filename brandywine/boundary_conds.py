import numpy as np
import brandywine.conservative_vars as cv

def inviscid_wall0(uvec:np.ndarray):
    return np.array([
        uvec[cv.Index.RHO], # densities equal
        -uvec[cv.Index.RHOU], # vel equal and opposite
        uvec[cv.Index.E] # pressures equal
    ])