import taichi as ti

from src.BEM_functions.single_layer import AbstractSingleLayer
from src.BEM_functions.utils import get_gaussion_integration_points_and_weights


@ti.data_oriented
class SingleLayer2d(AbstractSingleLayer):
    rank = 2

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(SingleLayer2d, self).__init__(BEM_manager, *args, **kwargs)

        self._GaussQR = self._BEM_manager.get_GaussQR()

        self._ti_dtype = self._BEM_manager._ti_dtype
        self._np_dtype = self._BEM_manager._np_dtype
        self._n = self._BEM_manager._n

        self.num_of_Dirichlets = self._BEM_manager.get_num_of_Dirichlets()
        self.num_of_Neumanns = self._BEM_manager.get_num_of_Neumanns()
        self.num_of_vertices = self._BEM_manager.get_num_of_vertices()
        self.num_of_panels = self._BEM_manager.get_num_of_panels()
        self._Vmat = None
    
    @ti.kernel
    def forward(self):
        pass
