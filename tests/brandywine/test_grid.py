import numpy as np
import brandywine.grid as grd

class TestGrid1D:
    start=-10
    stop=10
    ncells=200
    x = np.linspace(start, stop, ncells+1)
    grid = grd.Grid1D(x)

    def test_ncells(self):
        assert self.grid.ncells == self.ncells

    def test_nvertices(self):
        assert self.grid.nvertices == self.ncells+1

    def test_dx(self):
        np.testing.assert_allclose(self.grid.dx, np.zeros(self.ncells)+0.1)

    def test_cc(self):
        np.testing.assert_allclose(self.grid.cc, np.linspace(-9.95,9.95, 200))