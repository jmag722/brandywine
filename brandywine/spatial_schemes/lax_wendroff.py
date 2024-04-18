import numpy as np
import brandywine.conservative_vars as cv
from brandywine.spatial_schemes._iflux_utils import inviscid_flux_vector as E
from brandywine.spatial_schemes._iflux_utils import inviscid_flux_jacobian as A
from brandywine.spatial_schemes._iflux_utils import artificial_dissipation as D
from brandywine.spatial_schemes._iflux_utils import pressure_sensor as d2p_4p

def spatial_derivative(U:cv.ConservativeVars, index:int, gam:float,
                       dx:float, dt:float, **kwargs):
    pterm_p = d2p_4p(Uim1=U[index-1], Ui=U[index], Uip1=U[index+1], gam=gam)
    pterm_m = 0.
    if index > 1:
        pterm_m = d2p_4p(Uim1=U[index-2], Ui=U[index-1], Uip1=U[index], gam=gam)
    return 1/dx * (
        flux_interface(Ui=U[index], Uip1=U[index+1], gam=gam, dx=dx, dt=dt,
                       oscillation_strength=pterm_p, **kwargs)
      - flux_interface(Ui=U[index-1], Uip1=U[index], gam=gam, dx=dx, dt=dt,
                       oscillation_strength=pterm_m, **kwargs)
  )

def flux_interface(Ui:np.ndarray, Uip1:np.ndarray, gam:float, dx:float, dt:float, **kwargs):
    """
    Flux interface for Lax-Wendroff scheme.

    E_{i+1/2} (LW) = E_{i+1/2} - D_{i+1/2}*(U_{i+1/2} - U_i) - 0.5dt*A_{i+1/2}dE/dx_{i+1/2}

    Parameters
    ----------
    Ui : np.ndarray
        U_i
    Uip1 : np.ndarray
        U_{i+1}
    gam : float
        ratio of specific heats
    dx : float
        cell size
    dt : float
        time step

    Returns
    -------
    np.ndarray
        E_{i+1/2} (LW)
    """    
    return (
          0.5*(E(Uip1, gam) + E(Ui, gam))
        # if 0.5* term multiplied to D, default epsilon should increase
        - D(Ui=Ui, Uip1=Uip1, gam=gam, **kwargs) * (Uip1 - Ui)
        - 0.5*dt * A(0.5*(Ui + Uip1), gam) @ (E(Uip1, gam) - E(Ui, gam))/dx
    )