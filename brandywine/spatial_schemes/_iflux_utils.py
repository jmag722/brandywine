import numpy as np

import brandywine.conservative_vars as cv
import brandywine.state_equations as eos

def inviscid_flux_vector(U:np.ndarray, gam:float):
    return np.array([
        cv.momentum(U),
        2*cv.kinetic_energy(U) + eos.pressure(cv.total_energy(U), cv.kinetic_energy(U), gam),
        ( cv.total_energy(U) + eos.pressure(cv.total_energy(U), cv.kinetic_energy(U), gam) ) * cv.velocity(U)
    ])