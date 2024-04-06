import numpy as np

def runge_kutta1(U0:np.ndarray, dt:float, spatial_derivative:np.ndarray):
    """
    Runge Kutta 1, aka forward Euler method.

    Parameters
    ----------
    U0 : np.ndarray
        Conservative variable array
    dt : float
        timestep
    spatial_derivative : np.ndarray
        Spatial derivative, -(E_{i+0.5} - E_{i-0.5})/dx

    Returns
    -------
    np.ndarray
        Solution at new timestep
    """    
    return U0 - dt*spatial_derivative