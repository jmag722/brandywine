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

    def test_ntotcells(self):
        assert self.grid.ntotcells == self.ncells + grd.Grid1D.nghosts

    def test_ntotvertices(self):
        assert self.grid.ntotvertices == self.ncells+1 + grd.Grid1D.nghosts

    def test_dx(self):
        np.testing.assert_allclose(self.grid.dx, np.zeros(self.ncells+2)+0.1)

    def test_cc(self):
        np.testing.assert_allclose(self.grid.cc, np.linspace(-10.05, 10.05, 202))