import taichi as ti

from src.BEM_functions.hypersingular_layer import AbstractHypersingularLayer


@ti.data_oriented
class HypersingularLayer2d(AbstractHypersingularLayer):
    rank = 2

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(HypersingularLayer2d, self).__init__(BEM_manager, *args, **kwargs)
        self._GaussQR = self._BEM_manager.get_GaussQR()

        self._ti_dtype = self._BEM_manager._ti_dtype
        self._np_dtype = self._BEM_manager._np_dtype
        self._n = self._BEM_manager._n
    
    @ti.func
    def forward(self):
        pass
