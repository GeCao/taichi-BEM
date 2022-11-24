from abc import ABC, abstractmethod
import taichi as ti


@ti.data_oriented
class AbstractDoubleLayer(ABC):
    def __init__(self, BEM_manager, *args, **kwargs,):
        self._BEM_manager = BEM_manager

    @property
    @abstractmethod
    def rank(self) -> int:
        ...

    @abstractmethod
    @ti.func
    def forward(self):
        ...
