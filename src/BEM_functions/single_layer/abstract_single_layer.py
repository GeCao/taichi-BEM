from abc import ABC, abstractmethod
import taichi as ti


@ti.data_oriented
class AbstractSingleLayer(object):
    def __init__(self, BEM_manager, *args, **kwargs,):
        self._BEM_manager = BEM_manager
    
    @ti.func
    def G(self, x, y, k, sqrt_n):
        return self._BEM_manager.G(x, y, k, sqrt_n)
