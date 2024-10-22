import abc

import numpy as np
import brandywine.conservative_vars as cv

class InviscidFluxInterface(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def names(cls) -> list[str]:
        raise NotImplementedError("Child class must share its ID names.")
    
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    def spatial_derivative(self,
        U:cv.ConservativeVars, index:int, gam:float, dx:float, dt:float):
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
        return 1./dx * (
            self.flux_interface(Ui=U[index], Uip1=U[index+1], gam=gam,
                                dx=dx, dt=dt)
            - self.flux_interface(Ui=U[index-1], Uip1=U[index], gam=gam,
                                  dx=dx, dt=dt)
        )

    @abc.abstractmethod
    def flux_interface(
        self, Ui:np.ndarray, Uip1:np.ndarray, gam:float,
        dx:float, dt:float, **kwargs):
        raise NotImplementedError("Child class must implement its own "
                                  "flux interface method.")
