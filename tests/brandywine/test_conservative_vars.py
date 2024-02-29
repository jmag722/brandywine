import numpy as np
import brandywine.conservative_vars as cv

class TestConservativeVars:
    rho = np.random.rand(10) + 1
    u = np.zeros_like(rho)+3
    rhou = rho*u
    e = np.random.rand(10)*10+50
    U = cv.ConservativeVars(rho, rhou, e)

    def test_shape(self):
        assert self.U.shape == (self.rho.size, cv.Index.SIZE)

    def test_size(self):
        assert self.U.size == self.rho.size

    def test_r(self):
        np.testing.assert_equal(self.U.r, self.rho)
    
    def test_ru(self):
        np.testing.assert_equal(self.U.ru, self.rhou)

    def test_ke(self):
        np.testing.assert_allclose(self.U.ke, 0.5*self.rho*self.u**2)

    def test_u(self):
        np.testing.assert_allclose(self.U.u, self.u)

    def test_e(self):
        np.testing.assert_equal(self.U.e, self.e)
    
    def test_getitem(self):
        for i in range(self.rho.size):
            np.testing.assert_equal(
                self.U[i],
                np.array([self.rho[i], self.rhou[i], self.e[i]])
            )

    def test_setitem(self):
        self.U[4] = np.array([0,1,0])
        for i in range(self.rho.size):
            if i != 4:
                np.testing.assert_equal(
                    self.U[i],
                    np.array([self.rho[i], self.rhou[i], self.e[i]])
                )
            else:
                np.testing.assert_equal(self.U[i], np.array([0,1,0]))
