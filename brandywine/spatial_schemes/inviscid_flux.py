from enum import IntEnum, unique
from typing import Callable
import brandywine.conservative_vars as cv
import brandywine.spatial_schemes.lax_friedrich as lax

@unique
class InviscidFluxId(IntEnum):
    LAX_FRIEDRICH = 0

_inviscid_flux_str2id = {
    "lax": InviscidFluxId.LAX_FRIEDRICH,
    "lax-friedrich": InviscidFluxId.LAX_FRIEDRICH,
    "lf": InviscidFluxId.LAX_FRIEDRICH
}
_inviscid_flux_fmap = {
    InviscidFluxId.LAX_FRIEDRICH: lax.flux_interface,
}

def get_inviscid_scheme(flux_name:str):
    if flux_name.lower() not in _inviscid_flux_str2id:
        raise KeyError(f"Inviscid flux `{flux_name}` is not supported.")
    flux_id = _inviscid_flux_str2id[flux_name.lower()]
    return _inviscid_flux_fmap[flux_id]

def spatial_derivative(U:cv.ConservativeVars, index:int, gam:float,
                       dx:float, dt:float, flux_func:Callable):
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
  flux_func : Callable
      dynamically-called inviscid flux vector at the interface (scheme dependent)

  Returns
  -------
  np.ndarray
      spatial derivative of the inviscid flux
  """
  return 1/dx * (
      flux_func(Ui=U[index], Uip1=U[index+1], gam=gam, dx=dx, dt=dt)
    - flux_func(Ui=U[index-1], Uip1=U[index], gam=gam, dx=dx, dt=dt)
  )