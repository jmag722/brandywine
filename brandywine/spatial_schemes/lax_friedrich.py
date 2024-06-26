import numpy as np
import brandywine.conservative_vars as cv
from brandywine.spatial_schemes._iflux_utils import inviscid_flux_vector as E

def spatial_derivative(U:cv.ConservativeVars, index:int, gam:float,
                       dx:float, dt:float):
  """
  Spatial derivative: partialE/partialx = 1/dx(E_{i+0.5} - E_{i-0.5})

  Parameters
  ----------
  U : cv.ConservativeVars
      conservative variable array
  index : int
      position in solution grid
  gam : float
      ratio of specific heats
  dx : float
      grid size
  dt : float
      timestep

  Returns
  -------
  np.ndarray
      spatial derivative of the inviscid flux
  """
  return 1/dx * (
      flux_interface(Ui=U[index], Uip1=U[index+1], gam=gam, dx=dx, dt=dt)
    - flux_interface(Ui=U[index-1], Uip1=U[index], gam=gam, dx=dx, dt=dt)
  )

def flux_interface(Ui:np.ndarray, Uip1:np.ndarray, gam:float, dx:float, dt:float):
    """
    Compute the inviscid flux at the cell interface, E_{i+0.5}, for the Lax
    Friedrich (Lax) scheme.

    E_{i+0.5} (Lax) = (E_i + E_{i+1})/2 - dx/dt(U_{i+1} - U_i)/2

    Parameters
    ----------
    Ui : np.ndarray
        ith conservative variable
    Uip1 : np.ndarray
        i+1 conservative variable
    gam : float
        ratio of specific heats
    dx : float
        distance between cells
    dt : float
        time increment

    Returns
    -------
    np.ndarray
        inviscid flux vector at cell interface, E_{i+0.5}
    """    
    return 0.5*(E(Ui, gam) + E(Uip1, gam)) - 0.5*dx/dt * (Uip1 - Ui)