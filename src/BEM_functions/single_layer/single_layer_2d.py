import taichi as ti

from src.BEM_functions.single_layer import AbstractSingleLayer
from src.BEM_functions.utils import get_gaussion_integration_points_and_weights


@ti.data_oriented
class SingleLayer2d(AbstractSingleLayer):
    rank = 2

    def __init__(self, BEM_manager, mesh_manager, *args, **kwargs,):
        super(SingleLayer2d, self).__init__(BEM_manager, mesh_manager, *args, **kwargs)

        self._GaussQR = self._BEM_manager._GaussQR
        self._Q = self._mesh_manager._Q  # Number of local shape functions

        num_of_panels = len(self._mesh_manager.panels) // self._mesh_manager.dim
        self._Vmat = ti.field(dtype=self._mesh_manager._ti_dtype, shape=(num_of_panels, num_of_panels))
    
    @ti.kernel
    def forward(self):
        dim = 2
        num_of_panels = len(self._mesh_manager.panels) // dim

        Gauss_points, Gauss_weights = get_gaussion_integration_points_and_weights(
            N=self._GaussQR,
            ti_type=self._mesh_manager._ti_dtype,
            np_type=self._mesh_manager._np_dtype
        )  # Points from -1 to 1, weights can be sum up to 2

        for i in ti.static(range(num_of_panels)):
            for j in ti.static(range(num_of_panels)):
                # Construct a local matrix
                if self._mesh_manager.is_same_panel(i, j):
                    # Coincide
                    integrand1 = 0.0
                    for ii in ti.static(range(self._Q)):
                        for jj in ti.static(range(self._Q)):
                            for iii in ti.static(range(self._GaussQR)):
                                for jjj in ti.static(range(self._GaussQR)):
                                    pass
                elif self._mesh_manager.is_common_vertex_panel(i, j):
                    # neighbor
                    pass
                else:
                    # Far away
                    pass
                # Fill local matrix to global
