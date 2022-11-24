import taichi as ti
import math

from src.parametrizations.abstract_parametrization import AbstractParametrization


@ti.data_oriented
class ParametrizedPolynomial(AbstractParametrization):
    def __init__(self, N: int, dim: int, *args, **kwargs,):
        super(ParametrizedPolynomial, self).__init__(*args, **kwargs)
        self._N = N
        self._dim = dim
        self._coeff = ti.field(self._ti_dtype, shape=(self._N, self._dim))
    
    @ti.func
    def eval(self, t: float) -> float:
        """
        Args:
            t (float): parameter in [-1, 1]

        Returns:
            [float]: the evaluation of polynomial functions
        """
        dim = self._dim

        x = t * (self._tmax - self._tmin) / 2.0 + (self._tmax + self._tmin) / 2.0
        result = [0.0] * dim
        for i, j in self._coeff:
            result[j] += self._coeff[i, j] * math.pow(x, i)
        
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

        x = t * (self._tmax - self._tmin) / 2.0 + (self._tmax + self._tmin) / 2.0
        derivative = [0.0] * dim
        for i, j in self._coeff:
            if i > 0:
                derivative[j] += self._coeff[i, j] * i * math.pow(x, i - 1)
        
        for j in range(dim):
            derivative[j] *= (self._tmax - self._tmin) / 2.0
        
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

        x = t * (self._tmax - self._tmin) / 2.0 + (self._tmax + self._tmin) / 2.0
        double_derivative = [0.0] * dim
        for i, j in self._coeff:
            if i > 1:
                double_derivative[j] += self._coeff[i, j] * i * (i - 1) * math.pow(x, i - 2)
        
        for j in range(dim):
            double_derivative[j] *= (self._tmax - self._tmin) / 2.0 * (self._tmax - self._tmin) / 2.0
        
        return double_derivative
