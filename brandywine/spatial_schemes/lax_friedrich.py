import numpy as np
from brandywine.spatial_schemes._iflux_utils import inviscid_flux_vector as E

import brandywine.conservative_vars as cv

def flux(U:cv.ConservativeVars, index:int,
         gam:float, dx:float, dt:float):
  """
  Lax-Friedrich or Lax inviscid flux scheme.

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
      inviscid flux
  """    
  return 1/dx * (
      _E_lax_face(Ui=U[index], Uip1=U[index+1], gam=gam, dx=dx, dt=dt)
    - _E_lax_face(Ui=U[index-1], Uip1=U[index], gam=gam, dx=dx, dt=dt)
  )

def _E_lax_face(Ui:np.ndarray, Uip1:np.ndarray, gam:float, dx:float, dt:float):
    return 0.5*(E(Ui, gam) + E(Uip1, gam)) - 0.5*dx/dt * (Uip1 - Ui)