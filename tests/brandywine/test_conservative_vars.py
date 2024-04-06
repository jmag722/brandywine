import numpy as np
import brandywine.conservative_vars as cv        

class TestConservativeVars:
    density = np.random.rand(10) + 1
    velocities = np.random.rand(10)+30
    momentum = density*velocities
    kinetic_energies = 0.5*density*velocities**2
    total_energy = np.random.rand(10)*10+50
    U = cv.ConservativeVars(density, momentum, total_energy)

    def test_property_density(self):
        np.testing.assert_equal(self.U.density, self.density)
    
    def test_property_momentum(self):
        np.testing.assert_equal(self.U.momentum, self.momentum)

    def test_property_total_energy(self):
        np.testing.assert_equal(self.U.total_energy, self.total_energy)
    
    def test_getitem(self):
        for i in range(self.density.size):
            np.testing.assert_equal(
                self.U[i],
                np.array([self.density[i], self.momentum[i], self.total_energy[i]])
            )

    def test_setitem(self):
        density = np.random.rand(10) + 1
        momentum = np.random.rand(10)+30
        total_energy = np.random.rand(10)*10+50
        U = cv.ConservativeVars(density, momentum, total_energy)
        U[4] = np.array([0,1,0])
        for i in range(density.size):
            if i != 4:
                np.testing.assert_equal(
                    U[i],
                    np.array([density[i], momentum[i], total_energy[i]])
                )
            else:
                np.testing.assert_equal(U[i], np.array([0,1,0]))

    def test_density(self):
        np.testing.assert_equal(cv.density(self.U[3]), self.density[3])
        np.testing.assert_equal(np.apply_along_axis(cv.density, 1, self.U),
                                self.density)

    def test_momentum(self):
        np.testing.assert_equal(cv.momentum(self.U[3]), self.momentum[3])
        np.testing.assert_equal(np.apply_along_axis(cv.momentum, 1, self.U),
                                self.momentum)

    def test_kinetic_energy(self):
        np.testing.assert_allclose(cv.kinetic_energy(self.U[3]),
                                   self.kinetic_energies[3])
        np.testing.assert_allclose(np.apply_along_axis(cv.kinetic_energy, 1, self.U),
                                   self.kinetic_energies)

    def test_total_energy(self):
        np.testing.assert_equal(cv.total_energy(self.U[3]),
                                self.total_energy[3])
        np.testing.assert_equal(np.apply_along_axis(cv.total_energy, 1, self.U),
                                self.total_energy)

    def test_velocity(self):
        np.testing.assert_allclose(cv.velocity(self.U[5]), self.velocities[5])
        np.testing.assert_allclose(np.apply_along_axis(cv.velocity, 1, self.U),
                                   self.velocities)