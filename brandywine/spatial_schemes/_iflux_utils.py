import numpy as np

import brandywine.conservative_vars as cv
import brandywine.state_equations as eos

def inviscid_flux_vector(U:np.ndarray, gam:float):
    """
    Flux vector E, only taking into account inviscid components.
        Relation is [rho*u, rho*u^2+p, (e+p)*u]^T, where e is the total
        energy density, p is pressure, u is velocity, and rho is density

    Parameters
    ----------
    U : np.ndarray
        Conservative variable vector
    gam : float
        ratio of specific heats

    Returns
    -------
    np.ndarray
        inviscid flux vector
    """
    # saving local variables to avoid recomputing over and over,
    # because for 1D Euler memory is not an issue
    ke = cv.kinetic_energy(U)
    e = cv.total_energy(U)
    p = eos.pressure(e, ke, gam)
    return np.array([
        cv.momentum(U),
        2*ke + p,
        (e + p) * cv.velocity(U)
    ])