import taichi as ti

from src.BEM_functions.double_layer import AbstractDoubleLayer


@ti.data_oriented
class DoubleLayer2d(AbstractDoubleLayer):
    rank = 2

    def __init__(self, BEM_manager, mesh_manager, *args, **kwargs,):
        super(DoubleLayer2d, self).__init__(BEM_manager, mesh_manager, *args, **kwargs)
        self._GaussQR = self._BEM_manager._GaussQR
        self._Q = self._mesh_manager._Q  # Number of local shape functions

        self._ti_dtype = self._mesh_manager._ti_dtype
        self._np_dtype = self._mesh_manager._np_dtype
    
    @ti.func
    def forward(self):
        pass
