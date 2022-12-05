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

    @property
    @abstractmethod
    def rank(self) -> int:
        ...

    @abstractmethod
    @ti.func
    def forward(self):
        ...
