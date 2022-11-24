import taichi as ti
import numpy as np

from src.BEM_functions.rhs_constructor import AbstractRHSConstructor
from src.BEM_functions.utils import get_gaussion_integration_points_and_weights
from src.managers.mesh_manager import CellType, KernelType, PanelsRelation


@ti.data_oriented
class RHSConstructor3d(AbstractRHSConstructor):
    rank = 3

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(RHSConstructor3d, self).__init__(BEM_manager, *args, **kwargs)

        self._GaussQR = self._BEM_manager._GaussQR
        self._Q = self._BEM_manager._Q  # Number of local shape functions

        self._ti_dtype = self._BEM_manager._ti_dtype
        self._np_dtype = self._BEM_manager._np_dtype
        self._kernel_type = self._BEM_manager._kernel_type
        self._k = self._BEM_manager._k
        self._n = self._BEM_manager._n

        self.num_of_Dirichlets = self._BEM_manager.get_num_of_Dirichlets()
        self.num_of_Neumanns = self._BEM_manager.get_num_of_Neumanns()
        self.num_of_vertices = self._BEM_manager.get_num_of_vertices()
        self.num_of_panels = self._BEM_manager.get_num_of_panels()
        self.analyical_function_Dirichlet = self._BEM_manager.analyical_function_Dirichlet
        self.analyical_function_Neumann = self._BEM_manager.analyical_function_Neumann
        self._vert_g_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_vertices,))
        self._vert_f_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_vertices,))
        self._set_boundaries()
        self._gvec = None
        if self.num_of_Dirichlets > 0:
            self._gvec = ti.Vector.field(
                self._n,
                dtype=self._ti_dtype,
                shape=(self.num_of_Dirichlets * self._Q,)
            )
        self._fvec = None
        if self.num_of_Neumanns > 0:
            self._fvec = ti.Vector.field(
                self._n,
                dtype=self._ti_dtype,
                shape=(self.num_of_Neumanns * self._Q,)
            )

        self.m_mats_coincide = ti.Matrix.field(4, 4, dtype=self._ti_dtype, shape=(3,))
        np_m_mats_coincide = np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, -1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0]
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, -1.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                [
                    [1.0, 0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0, -1.0]
                ],
            ],
            dtype=self._np_dtype
        )
        self.m_mats_coincide.from_numpy(np_m_mats_coincide)

        self.Gauss_points_1d = ti.field(self._ti_dtype, shape=(self._GaussQR, ))
        self.Gauss_weights_1d = ti.field(self._ti_dtype, shape=(self._GaussQR, ))
        np_Gauss_points_1d, np_Gauss_weights_1d = get_gaussion_integration_points_and_weights(
            N=self._GaussQR,
            np_type=self._np_dtype
        )
        self.Gauss_points_1d.from_numpy(np_Gauss_points_1d)
        self.Gauss_weights_1d.from_numpy(np_Gauss_weights_1d)

    @ti.kernel
    def _set_boundaries(self):
        for i in range(self.num_of_vertices):
            self._vert_g_boundary[i] = self.analyical_function_Dirichlet(self._BEM_manager.get_vertice(i))
            # TODO:
            self._vert_f_boundary[i] = 0 * self._vert_f_boundary[i]  # self.analyical_function_Neumann(self._BEM_manager.get_vertice(i), self._BEM_manager.get_vert_normal(i))
    
    @ti.func
    def get_g_vec(self):
        """
        Get the rhs vector in Dirichlet Region
        """
        return self._gvec

    @ti.func
    def get_f_vec(self):
        """
        Get the rhs vector in Neumann Region
        """
        return self._fvec

    def kill(self):
        self.num_of_Dirichlets = 0
        self.num_of_Neumanns = 0
        self._gvec = None
        self._fvec = None
        self.m_mats_coincide = None
        self.Gauss_points_1d = None
        self.Gauss_weights_1d = None
    
    @ti.func
    def surface_grad(self, x1, x2, x3, u1, u2, u3, normal, area):
        # Why 1/2 ? This (grad_p) * (area) = (du / height) * (height * edge / 2) = (du * edge / 2)
        gradu_integral = (
            u1 * normal.cross(x3 - x2) +
            u2 * normal.cross(x1 - x3) +
            u3 * normal.cross(x2 - x1)
        ) / 2.0  #  grad_p times face area
        grad_u = gradu_integral / area

        return grad_u

    @ti.func
    def surface_grad_phi(self, x1, x2, x3, normal, area, i: int):
        """
        Providing x (r1, r2) = (1 - r1) * x1 + (r1 - r2) * x2 + r2 * x3

        case i == 0:
            phi_0 = b_N^0 = 1 - r1
        case i == 1:
            phi_1 = b_N^0 = r1 - r2
        case i == 2:
            phi_2 = b_N^0 = r2
        """
        u1, u2, u3 = 0.0, 0.0, 0.0
        if i == 0:
            u1, u2, u3 = 1.0, 0.0, 0.0
        elif i == 1:
            u1, u2, u3 = 0.0, 1.0, 0.0
        elif i == 2:
            u1, u2, u3 = 0.0, 0.0, 1.0
        else:
            print("For 3D case, a shape function only accept 0, 1, or 2, not {}".format(i))
        
        grad_u = self.surface_grad(x1, x2, x3, u1, u2, u3, normal, area)
        return grad_u
    
    @ti.func
    def interplate_from_unit_triangle_to_general(self, r1, r2, x1, x2, x3):
        """
        r2
         ^
        1|                                      x2
         |   /|                                 /|
         |  / |                                / |
         | /  |                      ->       /  |
         |/   |                              /   |
        0|----|1--->r1                    x3/____|x1
         0
         
         - How to project a unit triangle (x1, x2, x3) to a general one?
         - If we choose (0, 0)->x1, (1, 0)->x2, (1, 1)->x3
         - x (r1, r2) = (1 - r1) * x1 + (r1 - r2) * x2 + r2 * x3
        """
        return (1 - r1) * x1 + (r1 - r2) * x2 + r2 * x3
    
    @ti.func
    def shape_function(self, r1, r2, i: int):
        """
        Providing x (r1, r2) = (1 - r1) * x1 + (r1 - r2) * x2 + r2 * x3

        case i == 0:
            b_N^0 = 1 - r1
        case i == 1:
            b_N^0 = r1 - r2
        case i == 2:
            b_N^0 = r2
        """
        return 1 - r1 if i == 0 else (r1 - r2 if i == 1 else r2)
    
    @ti.func
    def G(self, x, y):
        Gxy = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
        if ti.static(self._kernel_type == int(KernelType.LAPLACE)):
            Gxy.x = (1.0 / 4.0 / ti.math.pi) / (x - y).norm()
        elif ti.static(self._kernel_type == int(KernelType.HELMHOLTZ)):
            distance = (x - y).norm()
            Gxy = (1.0 / 4.0 / ti.math.pi) / distance * ti.math.cexp(
                ti.Vector([0.0, self._k * distance], self._ti_dtype)
            )
        
        return Gxy
    
    @ti.func
    def integrate_on_single_triangle(
        self,
        triangle_x: int,
        basis_function_index_x: int,
        vec_type: int,
    ):
        """
        Reference: S. A. Sauter and C. Schwab, Boundary Element Methods, Springer Ser. Comput. Math. 39, Springer-Verlag, Berlin, 2011.
        https://link.springer.com/content/pdf/10.1007/978-3-540-68093-2.pdf
        See this algorithm from chapter 5.2.1

        Get Integration on coincide triangles, where
        int_{Tau_x} int_{Tau_x} (func) dx dy
            = (2 * area_x) * (2 * area_y) *           int_{unit_triangle}    int_{unit_triangle}     (func) dx             dy
            = (2 * area_x) * (2 * area_y) * sum_{6} * int_{0->1} int_{0->w1} int_{0->w2} int_{0->w3} (func) d(w1) d(w2)    d(w3) d(w4)
            = (2 * area_x) * (2 * area_y) * sum_{6} * int_{0->1} int_{0->1}  int_{0->1}  int_{0->1}  (func) d(xsi) d(eta1) d(eta2) d(eta3)
        
         0

         ============= TO =============
          r2
         ^
        1|                                      x2
         |   /|                                 /|
         |  / |                                / |
         | /  |                      ->       /  |
         |/   |                              /   |
        0|----|1--->r1                    x3/____|x1
         0
         
         - How to project a unit triangle (x1, x2, x3) to a general one?
         - If we choose (0, 0)->x1, (1, 0)->x2, (1, 1)->x3
         - x (r1, r2) = (1 - r1) * x1 + (r1 - r2) * x2 + r2 * x3
        """
        integrand = ti.Vector([0.0 for i in range(self._n)])

        x1 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_x + 0)
        x2 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_x + 1)
        x3 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_x + 2)
        area_x = self._BEM_manager.get_panel_area(triangle_x)
        normal_x = self._BEM_manager.get_panel_normal(triangle_x)

        vec_type = self._BEM_manager.get_panel_type(triangle_x)
        
        for iii in range(self._GaussQR * self._GaussQR):
            # Generate number(r1, r2)
            r1_x = self.Gauss_points_1d[iii // self._GaussQR]
            r2_x = self.Gauss_points_1d[iii % self._GaussQR] * r1_x

            # Scale your weight
            weight_x = self.Gauss_weights_1d[iii // self._GaussQR] * self.Gauss_weights_1d[iii % self._GaussQR] * (area_x * 2.0)

            # Get your final weight
            weight = weight_x

            x = self.interplate_from_unit_triangle_to_general(
                r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
            )
            jacobian = r1_x
            if vec_type == int(CellType.DIRICHLET):
                gx = self.analyical_function_Dirichlet(x)
                integrand += 0.5 * (
                    gx * self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                ) * weight * jacobian
            elif vec_type == int(CellType.NEUMANN):
                fx = self.analyical_function_Neumann(x, normal_x)
                integrand += 0.5 * (
                    fx * self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                ) * weight * jacobian
            else:
                print("The Cell type should only be Dirichlet or Neumann")
        
        return integrand
    
    @ti.kernel
    def forward(self):
        """
        ### Perspective from Parametric functions
        We have F(), and G() functions for BEM in this perspective.
        Usually, both of them requires to calculate a "evaluateShapeFunction" and a "pi_p.Derivative(t).norm()".
        They are the "phase functions" and "Jacobian" we have mentioned before.
        """
        if ti.static(self.num_of_Dirichlets > 0):
            self._gvec.fill(0)

            # += K * g
            self._BEM_manager.double_layer.apply_K_dot_boundary(
                vert_boundary=self._vert_g_boundary, result_vec=self._gvec, cell_type=int(CellType.DIRICHLET), add=True
            )
            # += -V * f
            self._BEM_manager.single_layer.apply_V_dot_boundary(
                vert_boundary=self._vert_f_boundary, result_vec=self._gvec, cell_type=int(CellType.DIRICHLET), add=False
            )
        
            for I in self._gvec:
                i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(I // self._Q)
                ii = I % self._Q

                self._gvec[I] += self.integrate_on_single_triangle(
                    triangle_x=i, basis_function_index_x=ii, vec_type=int(CellType.DIRICHLET)
                )

        if ti.static(self.num_of_Neumanns > 0):
            self._fvec.fill(0)

            # += K * f
            self._BEM_manager.double_layer.apply_K_dot_boundary(
                vert_boundary=self._vert_f_boundary, result_vec=self._fvec, cell_type=int(CellType.NEUMANN), add=True
            )
            # += W * g
            self._BEM_manager.hypersingular_layer.apply_W_dot_boundary(
                vert_boundary=self._vert_g_boundary, result_vec=self._fvec, cell_type=int(CellType.NEUMANN), add=True
            )

            for I in self._fvec:
                i = self._BEM_manager.map_local_Neumann_index_to_panel_index(I // self._Q)
                ii = I % self._Q

                self._fvec[I] += self.integrate_on_single_triangle(
                    triangle_x=i, basis_function_index_x=ii, vec_type=int(CellType.NEUMANN), add=True
                )
