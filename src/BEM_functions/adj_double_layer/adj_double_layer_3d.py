import taichi as ti

from src.BEM_functions.adj_double_layer import AbstractAdjDoubleLayer


@ti.data_oriented
class AdjDoubleLayer3d(AbstractAdjDoubleLayer):
    rank = 3

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(AdjDoubleLayer3d, self).__init__(BEM_manager, *args, **kwargs)
        self._BEM_manager = BEM_manager
    
    def kill(self):
        self._K1mat = None
        self.m_mats_coincide = None
    
    @ti.func
    def forward(self):
        pass
