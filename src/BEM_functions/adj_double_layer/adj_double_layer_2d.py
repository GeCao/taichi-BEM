import taichi as ti

from src.BEM_functions.adj_double_layer import AbstractAdjDoubleLayer


@ti.data_oriented
class AdjDoubleLayer2d(AbstractAdjDoubleLayer):
    rank = 2

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(AdjDoubleLayer2d, self).__init__(BEM_manager, *args, **kwargs)
        self._BEM_manager = BEM_manager
    
    @ti.func
    def forward(self):
        pass
