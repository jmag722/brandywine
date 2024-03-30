import numpy as np
import brandywine.conservative_vars as cv
import brandywine.state_equations as eos

def inviscid_flux_vector(U:np.ndarray, gam:float):
    return np.array([
        cv.momentum(U),
        2*cv.kinetic_energy(U) + eos.pressure(cv.total_energy(U), cv.kinetic_energy(U), gam),
        (cv.total_energy(U) + eos.pressure(cv.total_energy(U), cv.kinetic_energy(U), gam)) * cv.velocity(U)
    ])
E = inviscid_flux_vector

def flux_lax_friederich(U:cv.ConservativeVars, index:int, gam:float, dx:float, dt:float):
    return 1/dx * (E_lax_face(Ui=U[index], Uip1=U[index+1], gam=gam, dx=dx, dt=dt)
                   - E_lax_face(Ui=U[index-1], Uip1=U[index], gam=gam, dx=dx, dt=dt))

def E_lax_face(Ui:np.ndarray, Uip1:np.ndarray, gam:float, dx:float, dt:float):
    return 0.5*(E(Ui, gam) + E(Uip1, gam)) - 0.5*dx/dt * (Uip1 - Ui)