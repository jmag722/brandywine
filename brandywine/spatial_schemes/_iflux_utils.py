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

def inviscid_flux_jacobian(U:np.ndarray, gam:float):
    """
    Returns 3x3 matrix A=dE/dU, the Jacobian for 1D conservative variable U. 

    0 , 1 , 0
    -(3-\gamma)u^2/2 , (3-\gamma)u , \gamma-1
    (\gamma-1)u^3-\gamma*u*e/\rho , \gamma*e/\rho - 1.5(\gamma-1)u^2 , \gamma*u

    Parameters
    ----------
    U : np.ndarray
        conservative variable
    gam : float
        ratio of specific heats

    Returns
    -------
    np.ndarray
        Jacobian matrix
    """    
    u = cv.velocity(U)
    e = U[cv.Index.RHOE] / U[cv.Index.RHO] # total specific energy
    return np.array([
        [ 0, 1, 0 ],
        [ -0.5 * (3 - gam) * u**2, (3 - gam) * u, gam - 1 ],
        [ (gam - 1) * u**3 - gam * u * e,
          gam * e - 1.5 * (gam - 1) * u**2,
          gam * u ]
    ])

def artificial_dissipation(Ui:np.ndarray, Uip1:np.ndarray, gam:float, epsilon:float=0.3,
                           oscillation_strength:float=1.0):
    """
    Artificial dissipation D
    D_{i+1/2} = epsilon*oscillation_strength*(|u|+c)_1/2

    Source:
    MacCormack, R. W. (2014). Numerical computation of compressible and
    viscous flow. American Institute of Aeronautics and Astronautics, Inc.

    Original source:
    Baldwin, B. and MacCormack, R.: Interaction of strong-shock wave with turbulent
        boundary layer. Proceedings of the Fourth International Conference on Numerical
        Methods in Fluid Dynamics Lecture Notes in Physics, (Berlin Heidelberg), 1975.

    Parameters
    ----------
    Ui : np.ndarray
        U_i
    Uip1 : np.ndarray
        U_{i+1}
    gam : float
        ratio of specific heats gamma
    epsilon : float, optional
        term to remove unwanted numerical error, by default 0.3
    oscillation_strength : float, optional
        Oscillation strength term that is low when solution is smooth,
        by default 1.0 (first order).

    Returns
    -------
    np.ndarray
        D_{i+1/2}
    """    
    return (
        epsilon
        * oscillation_strength
        * 0.5*(  np.abs(cv.velocity(Ui))   + cv.sound_speed(Ui, gam)
               + np.abs(cv.velocity(Uip1)) + cv.sound_speed(Uip1, gam))
    )

def pressure_sensor(Uim1:np.ndarray, Ui:np.ndarray, Uip1:np.ndarray, gam:float):
    """
    Oscillation strength dependent on the pressure between cells, d2p/4p

    |p_{i+1} - 2p_i + p_{i-1}| / (p_{i+1} + 2p_i + p_{i-1})

    Parameters
    ----------
    Uim1 : np.ndarray
        U_{i-1}
    Ui : np.ndarray
        U_i
    Uip1 : np.ndarray
        U_{i+1}
    gam : float
        ratio of specific heats gamma

    Returns
    -------
    d2p/4p
        pressure strength term
    """    
    n = np.abs(cv.pressure(cvar=Uip1, gam=gam) - 2*cv.pressure(cvar=Ui, gam=gam)
             + cv.pressure(cvar=Uim1, gam=gam))
    d = cv.pressure(cvar=Uip1, gam=gam) + 2*cv.pressure(cvar=Ui, gam=gam) + cv.pressure(cvar=Uim1, gam=gam)
    return n / d