from abc import ABC, abstractmethod
import taichi as ti


@ti.data_oriented
class AbstractDoubleLayer(ABC):
    def __init__(self, BEM_manager, *args, **kwargs,):
        self._BEM_manager = BEM_manager
        self._sqrt_n = 1.0
    
    @ti.func
    def G(self, x, y, sqrt_n):
        return self._BEM_manager.G(x, y, sqrt_n)
    
    @ti.func
    def grad_G_y(self, x, y, normal_y, sqrt_n):
        return self._BEM_manager.grad_G_y(x, y, normal_y, sqrt_n)
    
    def set_sqrt_n(self, sqrt_n):
        self._sqrt_n = sqrt_n

    @property
    @abstractmethod
    def rank(self) -> int:
        ...

    @abstractmethod
    @ti.func
    def forward(self):
        ...
