import numpy as np
import brandywine.grid as grd

class TestGrid1D:
    start=-10
    stop=10
    ncells=200
    x = np.linspace(start, stop, ncells+1)
    grid = grd.Grid1D(x)

    def test_istart_cell(self):
        assert self.grid.istart_cell == 1
    def test_istart_vert(self):
        assert self.grid.istart_vert == 1
    
    def test_iend_cell(self):
        assert self.grid.iend_cell == 200
    def test_iend_vert(self):
        assert self.grid.iend_vert == 201

    def test_range_cells(self):
        assert list(self.grid.range_cells) == list(range(1, self.grid.iend_cell+1))

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