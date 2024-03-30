import numpy as np

def rk1(U0:np.ndarray, dt:float, spatial_derivative:np.ndarray):
    return U0 - dt*spatial_derivative