from abc import ABC, abstractmethod
import taichi as ti


@ti.data_oriented
class AbstractHypersingularLayer(ABC):
    def __init__(self, BEM_manager, *args, **kwargs,):
        self._BEM_manager = BEM_manager
    
    @ti.func
    def G(self, x, y, sqrt_n):
        return self._BEM_manager.G(x, y, sqrt_n)
    
    @ti.func
    def grad_G_y(self, x, y, normal_y, sqrt_n):
        return self._BEM_manager.grad_G_y(x, y, normal_y, sqrt_n)
    
    @ti.func
    def grad2_G_xy(self, x, y, curl_phix_dot_curl_phiy, sqrt_n):
        return self._BEM_manager.grad2_G_xy(x, y, curl_phix_dot_curl_phiy, sqrt_n)

    @property
    @abstractmethod
    def rank(self) -> int:
        ...

    @abstractmethod
    @ti.func
    def forward(self):
        ...
