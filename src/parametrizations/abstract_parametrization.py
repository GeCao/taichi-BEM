from abc import ABC, abstractmethod
import taichi as ti


@ti.data_oriented
class AbstractParametrization(ABC):
    def __init__(self, tmin, tmax, ti_dtype):
        self._tmin = tmin
        self._tmax = tmax
        self._ti_dtype = ti_dtype

    @abstractmethod
    @ti.func
    def forward(self):
        ...
