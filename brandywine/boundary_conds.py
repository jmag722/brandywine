import numpy as np
import brandywine.conservative_vars as cv

def inviscid_wall0(uvec:np.ndarray):
    return np.array([
        uvec[cv.Index.RHO], # density equal
        -uvec[cv.Index.MOMX], # vel equal and opposite
        uvec[cv.Index.TOTENERGY] # pressures equal
    ])