import taichi as ti
from typing import List

from src.parametrizations.abstract_parametrization import AbstractParametrization


@ti.data_oriented
class ParametrizedLine(AbstractParametrization):
    def __init__(self, dim: int, x1: List[float], x2: List[float], *args, **kwargs,):
        super(ParametrizedLine, self).__init__(*args, **kwargs)
        self._dim = dim
        if dim != len(x1) or dim != len(x2):
            raise RuntimeError("The dimension input is not consistent with the point dimension for Parametrized Lines")
        
        self._x1 = ti.field(self._ti_dtype, shape=(self._dim,))
        self._x2 = ti.field(self._ti_dtype, shape=(self._dim,))
        for j in ti.static(range(dim)):
            self._x1[j] = x1[j]
            self._x2[j] = x2[j]
    
    @ti.func
    def eval(self, t: float) -> float:
        """
        Args:
            t (float): parameter in [-1, 1]

        Returns:
            [float]: the evaluation of polynomial functions
        """
        dim = self._dim
        
        result = [0.0] * dim
        for j in ti.static(range(dim)):
            result[j] = t * (self._x2[j] - self._x1[j]) / 2.0 + (self._x2[j] + self._x1[j]) / 2.0
        
        return result

    @ti.func
    def derivative(self, t: float):
        """
        Args:
            t (float): parameter in [-1, 1]

        Returns:
            [float]: the derivative of polynomial functions
        """
        dim = self._dim

        derivative = [0.0] * dim
        for j in ti.static(range(dim)):
            derivative[j] = (self._x2[j] - self._x1[j]) / 2.0
        
        return derivative

    @ti.func
    def double_derivative(self, t: float):
        """
        Args:
            t (float): parameter in [-1, 1]

        Returns:
            [float]: the derivative of polynomial functions
        """
        dim = self._dim

        double_derivative = [0.0] * dim
        
        return double_derivative
