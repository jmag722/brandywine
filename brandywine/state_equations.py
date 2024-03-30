import numpy as np

def pressure(total_internal_e, kinetic_e, gam:float):
    return (gam-1) * (total_internal_e - kinetic_e)
    
def sound_speed(p, rho, gam:float):
    return np.sqrt(gam*p/rho)
    
def total_internal_energy(p, rho, u, gam:float):
    return p/(gam-1) + 0.5*rho*u*u