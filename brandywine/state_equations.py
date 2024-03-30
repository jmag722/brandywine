import numpy as np

def pressure(total_energy, kinetic_energy, gam:float):
    return (gam-1) * (total_energy - kinetic_energy)
    
def sound_speed(pressure, density, gam:float):
    return np.sqrt(gam*pressure/density)
    
def total_energy(pressure, kinetic_energy, gam:float):
    return pressure/(gam-1) + kinetic_energy