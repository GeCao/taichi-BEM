from abc import ABC, abstractmethod
import taichi as ti


@ti.data_oriented
class AbstractRHSConstructor(ABC):
    def __init__(self, BEM_manager, *args, **kwargs,):
        self._BEM_manager = BEM_manager
    
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
