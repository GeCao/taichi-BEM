from abc import ABC, abstractmethod
import taichi as ti


@ti.data_oriented
class AbstractAdjDoubleLayer(ABC):
    def __init__(self, BEM_manager, *args, **kwargs,):
        self._BEM_manager = BEM_manager
    
    @ti.func
    def G(self, x, y, k, sqrt_n):
        return self._BEM_manager.G(x, y, k, sqrt_n)
    
    @ti.func
    def grad_G_y(self, x, y, normal_y, k, sqrt_n):
        return self._BEM_manager.grad_G_y(x, y, normal_y, k, sqrt_n)
    
    @ti.func
    def interplate_from_unit_panel_to_general(self, r1, r2, x1, x2, x3):
        return self._BEM_manager.interplate_from_unit_panel_to_general(r1, r2, x1, x2, x3)
    
    @ti.func
    def shape_function(self, r1, r2, i: int):
        return self._BEM_manager.shape_function(r1, r2, i)

    @property
    @abstractmethod
    def rank(self) -> int:
        ...

    @abstractmethod
    @ti.func
    def forward(self):
        ...
