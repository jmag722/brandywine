import brandywine.conservative_vars as cv

def rk1(U0:cv.ConservativeVars, dt:float, spatial_derivative):
    return U0 - dt*spatial_derivative