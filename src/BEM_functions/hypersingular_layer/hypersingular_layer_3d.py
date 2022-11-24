import taichi as ti
import numpy as np

from src.BEM_functions.hypersingular_layer import AbstractHypersingularLayer
from src.BEM_functions.utils import get_gaussion_integration_points_and_weights
from src.managers.mesh_manager import CellType, KernelType, PanelsRelation


@ti.data_oriented
class HypersingularLayer3d(AbstractHypersingularLayer):
    rank = 3

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(HypersingularLayer3d, self).__init__(BEM_manager, *args, **kwargs)

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
        assert(self.num_of_Dirichlets + self.num_of_Neumanns > 0)
        self._Wmat = None
        if self.num_of_Neumanns > 0:
            self._Wmat = ti.Vector.field(
                self._n,
                dtype=self._ti_dtype,
                shape=(self.num_of_Neumanns * self._Q, self.num_of_Neumanns * self._Q)
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
    
    @ti.func
    def get_W_mat(self):
        return self._Wmat
    
    def kill(self):
        self.num_of_Dirichlets = 0
        self.num_of_Neumanns = 0
        self._Wmat = None
        self.m_mats_coincide = None
        self.Gauss_points_1d = None
        self.Gauss_weights_1d = None

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
    def surface_grad(self, x1, x2, x3, u1, u2, u3, normal, area):
        # Why 1/2 ? This (grad_p) * (area) = (du / height) * (height * edge / 2) = (du * edge / 2)
        gradu_integral = (
            u1 * normal.cross(x3 - x2, dim=-1) +
            u2 * normal.cross(x1 - x3, dim=-1) +
            u3 * normal.cross(x2 - x1, dim=-1)
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
            raise RuntimeError("For 3D case, a shape function only accept 0, 1, or 2, not {}".format(i))
        
        grad_u = self.surface_grad(x1, x2, x3, u1, u2, u3, normal, area)
        return grad_u
    
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
    def grad_G_y(self, x, y, normal_y):
        grad_Gy = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
        if ti.static(self._kernel_type == int(KernelType.LAPLACE)):
            # A float number will be returned finally
            grad_Gy.x = (1.0 / 4.0 / ti.math.pi) * (x - y).dot(normal_y) / ti.math.pow((x - y).norm(), 3)
        elif ti.static(self._kernel_type == int(KernelType.HELMHOLTZ)):
            # A 2d Vector will be returned finally, as the distinction of Rel & Img part
            distance = (x - y).norm()
            exp_vector = ti.math.cexp(
                ti.Vector([0.0, self._k * distance], self._ti_dtype)
            )
            chained_complex_vector = exp_vector - self._k * distance * ti.Vector([-exp_vector.y, exp_vector.x])
            grad_Gy = (1.0 / 4.0 / ti.math.pi) * (x - y).dot(normal_y) / ti.math.pow(distance, 3) * chained_complex_vector
        
        return grad_Gy
    
    @ti.func
    def grad2_G_xy(self, x, y, curl_phix, curl_phiy):
        return (1.0 / 4.0 / ti.math.pi) / (x - y).norm() * (curl_phix.dot(curl_phiy))
    
    @ti.func
    def integrate_on_two_panels(
        self,
        vert_boundary,
        triangle_x: int,
        triangle_y: int,
        basis_function_index_x: int,
        basis_function_index_y: int,
        panels_relation: int
    ):
        """
        Get Integration points and weights for a general triangle (Use Duffy Transform)
        ### Perspective from Duffy Transform
         y
         ^
        1|__________
         |          |
         |  *    *  |
         |          |     <- This refers GaussQR = 2, you will get GaussQR*GaussQR=4 points and 4 weights
         |  *    *  |
        0-----------|1--->x
         0

         ============= TO =============

         y
         ^
        1|
         |   /|
         |  / |
         | /  |           <- Points: (x, y) := (x, x * y)
         |/   |           <- Weights:   w  To  w
        0|----|1--->x     <- Jaobian:   1   To  x
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

         ==============================
        To sum up, a random number (t, s) as a sample on triangle, where t,s \in [-1, 1]
        r1, r2          = ...
        Sampled_point   = (1 - r1) * x1  +  (r1 - r2) * x2  +  r2 * x3
        Corres_weight   = w * 2 * Area
        Jacobian        = r1
        phase functions = r1, r1 - r2, r2

        !!!!!!!!!!!!!!
        However, if these two panels has overlaps, such like common edegs, common vertices, even the same panel.
        For instance, the same panel:
        Reference: S. A. Sauter and C. Schwab, Boundary Element Methods, Springer Ser. Comput. Math. 39, Springer-Verlag, Berlin, 2011.
        https://link.springer.com/content/pdf/10.1007/978-3-540-68093-2.pdf
        See this algorithm from chapter 5.2.1

        Get Integration on coincide triangles, where
        int_{Tau_x} int_{Tau_x} (func) dx dy
            = (2 * area_x) * (2 * area_y) *           int_{unit_triangle}    int_{unit_triangle}     (func) dx             dy
            = (2 * area_x) * (2 * area_y) * sum_{6} * int_{0->1} int_{0->w1} int_{0->w2} int_{0->w3} (func) d(w1) d(w2)    d(w3) d(w4)
            = (2 * area_x) * (2 * area_y) * sum_{6} * int_{0->1} int_{0->1}  int_{0->1}  int_{0->1}  (func) d(xsi) d(eta1) d(eta2) d(eta3)
        """
        integrand = ti.Vector([0.0 for i in range(self._n)])

        x1 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_x + 0)
        x2 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_x + 1)
        x3 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_x + 2)
        area_x = self._BEM_manager.get_panel_area(triangle_x)
        normal_x = self._BEM_manager.get_panel_normal(triangle_x)
        curl_phix = ti.math.cross(
            normal_x,
            self.surface_grad_phi(x1=x1, x2=x2, x3=x3, normal=normal_x, area=area_x, i=basis_function_index_x)
        )

        y1 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_y + 0)
        y2 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_y + 1)
        y3 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_y + 2)
        area_y = self._BEM_manager.get_panel_area(triangle_y)
        normal_y = self._BEM_manager.get_panel_normal(triangle_y)
        curl_phiy = ti.math.cross(
            normal_y,
            self.surface_grad_phi(x1=x1, x2=x2, x3=x3, normal=normal_y, area=area_y, i=basis_function_index_y)
        )

        g1 = vert_boundary[self._BEM_manager.get_vertice_index_from_flat_panel_index(3 * triangle_y + 0)]
        g2 = vert_boundary[self._BEM_manager.get_vertice_index_from_flat_panel_index(3 * triangle_y + 1)]
        g3 = vert_boundary[self._BEM_manager.get_vertice_index_from_flat_panel_index(3 * triangle_y + 2)]

        GaussQR2 = self._GaussQR * self._GaussQR
        GaussQR4 = GaussQR2 * GaussQR2
        
        for gauss_number in range(GaussQR4):
            # Generate number(xsi, eta1, eta2, eta3)
            iii = gauss_number // GaussQR2
            jjj = gauss_number % GaussQR2

            # Scale your weight
            weight_x = self.Gauss_weights_1d[iii // self._GaussQR] * self.Gauss_weights_1d[iii % self._GaussQR] * (area_x * 2.0)
            weight_y = self.Gauss_weights_1d[jjj // self._GaussQR] * self.Gauss_weights_1d[jjj % self._GaussQR] * (area_y * 2.0)
            # Get your final weight
            weight = weight_x * weight_y

            if panels_relation == int(PanelsRelation.SEPARATE):
                # Generate number(r1, r2) for panel x
                r1_x = self.Gauss_points_1d[iii // self._GaussQR]
                r2_x = self.Gauss_points_1d[iii % self._GaussQR] * r1_x
                # Generate number(r1, r2) for panel y
                r1_y = self.Gauss_points_1d[jjj // self._GaussQR]
                r2_y = self.Gauss_points_1d[jjj % self._GaussQR] * r1_y

                # Get your jacobian
                jacobian = r1_x * r1_y

                x = self.interplate_from_unit_triangle_to_general(
                    r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                )
                y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                )
                inner_val_y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=g1, x2=g2, x3=g3
                )

                integrand += (
                    -self.grad2_G_xy(x, y, curl_phix, curl_phiy) * inner_val_y * self.shape_function(r1_x, r2_x, i=basis_function_index_x) * self.shape_function(r1_y, r2_y, i=basis_function_index_y)
                ) * weight * jacobian
            elif panels_relation == int(PanelsRelation.COINCIDE):
                # Generate number(xsi, eta1, eta2, eta3)
                xsi = self.Gauss_points_1d[iii // self._GaussQR]
                eta1 = self.Gauss_points_1d[iii % self._GaussQR]
                eta2 = self.Gauss_points_1d[jjj // self._GaussQR]
                eta3 = self.Gauss_points_1d[jjj % self._GaussQR]

                # Get your jacobian
                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2

                # This algorithm includes 6 regions D1 ~ D6
                # By symmetic of kernel, we can simply compress it into 3 regions
                w = ti.Vector([xsi, xsi * eta1, xsi * eta1 * eta2, xsi * eta1 * eta2 * eta3])
                for iiii in range(self.m_mats_coincide.shape[0]):
                    xz = self.m_mats_coincide[iiii] @ w  # On unit triangle
                    
                    r1_x, r2_x = xz[0], xz[1]
                    r1_z, r2_z = xz[2], xz[3]
                    r1_y, r2_y = r1_x - r1_z, r2_x - r2_z

                    x = self.interplate_from_unit_triangle_to_general(
                        r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                    )
                    y = self.interplate_from_unit_triangle_to_general(
                        r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                    )
                    inner_val_x = self.interplate_from_unit_triangle_to_general(
                        r1=r1_x, r2=r2_x, x1=g1, x2=g2, x3=g3
                    )
                    inner_val_y = self.interplate_from_unit_triangle_to_general(
                        r1=r1_y, r2=r2_y, x1=g1, x2=g2, x3=g3
                    )

                    # D1, D3, D5
                    integrand += (
                        -self.grad2_G_xy(x, y, curl_phix, curl_phiy) * inner_val_y * self.shape_function(r1_x, r2_x, i=basis_function_index_x) * self.shape_function(r1_y, r2_y, i=basis_function_index_y)
                    ) * weight * jacobian

                    # D2, D4, D6
                    integrand += (
                        -self.grad2_G_xy(x, y, curl_phix, curl_phiy) * inner_val_x * self.shape_function(r1_y, r2_y, i=basis_function_index_x) * self.shape_function(r1_x, r2_x, i=basis_function_index_y)
                    ) * weight * jacobian
            elif panels_relation == int(PanelsRelation.COMMON_VERTEX):
                # Generate number(xsi, eta1, eta2, eta3)
                xsi = self.Gauss_points_1d[iii // self._GaussQR]
                eta1 = self.Gauss_points_1d[iii % self._GaussQR]
                eta2 = self.Gauss_points_1d[jjj // self._GaussQR]
                eta3 = self.Gauss_points_1d[jjj % self._GaussQR]

                # This algorithm includes 6 regions D1, D2
                # D1
                w = ti.Vector(
                    [xsi, xsi * eta1, xsi * eta2, xsi * eta2 * eta3]
                )
                r1_x, r2_x = w[0], w[1]
                r1_y, r2_y = w[2], w[3]
                x = self.interplate_from_unit_triangle_to_general(
                    r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                )
                y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                )
                inner_val_y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=g1, x2=g2, x3=g3
                )

                jacobian = xsi * xsi * xsi * eta2
                integrand += (
                    -self.grad2_G_xy(x, y, curl_phix, curl_phiy) * inner_val_y * self.shape_function(r1_x, r2_x, i=basis_function_index_x) * self.shape_function(r1_y, r2_y, i=basis_function_index_y)
                ) * weight * jacobian

                # D2
                w = ti.Vector(
                    [xsi * eta2, xsi * eta2 * eta3, xsi, xsi * eta1]
                )
                r1_x, r2_x = w[0], w[1]
                r1_y, r2_y = w[2], w[3]
                x = self.interplate_from_unit_triangle_to_general(
                    r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                )
                y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                )
                inner_val_y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=g1, x2=g2, x3=g3
                )

                jacobian = xsi * xsi * xsi * eta2
                integrand += (
                    -self.grad2_G_xy(x, y, curl_phix, curl_phiy) * inner_val_y * self.shape_function(r1_x, r2_x, i=basis_function_index_x) * self.shape_function(r1_y, r2_y, i=basis_function_index_y)
                ) * weight * jacobian
            elif panels_relation == int(PanelsRelation.COMMON_EDGE):
                # Generate number(xsi, eta1, eta2, eta3)
                xsi = self.Gauss_points_1d[iii // self._GaussQR]
                eta1 = self.Gauss_points_1d[iii % self._GaussQR]
                eta2 = self.Gauss_points_1d[jjj // self._GaussQR]
                eta3 = self.Gauss_points_1d[jjj % self._GaussQR]

                # This algorithm includes 6 regions D1 ~ D5
                # D1
                w = ti.Vector(
                    [xsi, -xsi * eta1 * eta2, xsi * eta1 * (1.0 - eta2), xsi * eta1 * eta3]
                )
                r1_x, r2_x = w[0], w[3]
                r1_y, r2_y = r1_x + w[1], w[2]
                x = self.interplate_from_unit_triangle_to_general(
                    r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                )
                y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                )
                inner_val_y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=g1, x2=g2, x3=g3
                )

                jacobian = xsi * xsi * xsi * eta1 * eta1
                integrand += (
                    -self.grad2_G_xy(x, y, curl_phix, curl_phiy) * inner_val_y * self.shape_function(r1_x, r2_x, i=basis_function_index_x) * self.shape_function(r1_y, r2_y, i=basis_function_index_y)
                ) * weight * jacobian

                # D2
                w = ti.Vector(
                    [xsi, -xsi * eta1 * eta2 * eta3, xsi * eta1 * eta2 * (1.0 - eta3), xsi * eta1]
                )
                r1_x, r2_x = w[0], w[3]
                r1_y, r2_y = r1_x + w[1], w[2]
                x = self.interplate_from_unit_triangle_to_general(
                    r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                )
                y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                )
                inner_val_y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=g1, x2=g2, x3=g3
                )

                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
                integrand += (
                    -self.grad2_G_xy(x, y, curl_phix, curl_phiy) * inner_val_y * self.shape_function(r1_x, r2_x, i=basis_function_index_x) * self.shape_function(r1_y, r2_y, i=basis_function_index_y)
                ) * weight * jacobian

                # D3
                w = ti.Vector(
                    [xsi * (1.0 - eta1 * eta2), xsi * eta1 * eta2, xsi * eta1 * eta2 * eta3, xsi * eta1 * (1.0 - eta2)]
                )
                r1_x, r2_x = w[0], w[3]
                r1_y, r2_y = r1_x + w[1], w[2]
                x = self.interplate_from_unit_triangle_to_general(
                    r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                )
                y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                )
                inner_val_y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=g1, x2=g2, x3=g3
                )

                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
                integrand += (
                    -self.grad2_G_xy(x, y, curl_phix, curl_phiy) * inner_val_y * self.shape_function(r1_x, r2_x, i=basis_function_index_x) * self.shape_function(r1_y, r2_y, i=basis_function_index_y)
                ) * weight * jacobian

                # D4
                w = ti.Vector(
                    [xsi * (1.0 - eta1 * eta2 * eta3), xsi * eta1 * eta2 * eta3, xsi * eta1, xsi * eta1 * eta2 * (1.0 - eta3)]
                )
                r1_x, r2_x = w[0], w[3]
                r1_y, r2_y = r1_x + w[1], w[2]

                x = self.interplate_from_unit_triangle_to_general(
                    r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                )
                y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                )
                inner_val_y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=g1, x2=g2, x3=g3
                )

                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
                integrand += (
                    -self.grad2_G_xy(x, y, curl_phix, curl_phiy) * inner_val_y * self.shape_function(r1_x, r2_x, i=basis_function_index_x) * self.shape_function(r1_y, r2_y, i=basis_function_index_y)
                ) * weight * jacobian

                # D5
                w = ti.Vector(
                    [xsi * (1.0 - eta1 * eta2 * eta3), xsi * eta1 * eta2 * eta3, xsi * eta1 * eta2, xsi * eta1 * (1.0 - eta2 * eta3)]
                )
                r1_x, r2_x = w[0], w[3]
                r1_y, r2_y = r1_x + w[1], w[2]

                x = self.interplate_from_unit_triangle_to_general(
                    r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                )
                y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                )
                inner_val_y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=g1, x2=g2, x3=g3
                )
                
                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
                integrand += (
                    -self.grad2_G_xy(x, y, curl_phix, curl_phiy) * inner_val_y * self.shape_function(r1_x, r2_x, i=basis_function_index_x) * self.shape_function(r1_y, r2_y, i=basis_function_index_y)
                ) * weight * jacobian
        
        return integrand
    
    @ti.kernel
    def forward(self):
        """
        ### Perspective from Parametric functions
        We have F(), and G() functions for BEM in this perspective.
        Usually, both of them requires to calculate a "evaluateShapeFunction" and a "pi_p.Derivative(t).norm()".
        They are the "phase functions" and "Jacobian" we have mentioned before.
        """
        dim = 3

        if ti.static(self.num_of_Neumanns > 0):
            self._Wmat.fill(0)

            for I, J in self._Wmat:
                i = self._BEM_manager.map_local_Neumann_index_to_panel_index(I // dim)
                j = self._BEM_manager.map_local_Neumann_index_to_panel_index(J // dim)
                ii = I % dim
                jj = J % dim

                # Construct a local matrix
                panels_relation = self._BEM_manager.get_panels_relation(i, j)
                self._Wmat[I, J] += self.integrate_on_two_panels(
                    vert_boundary=self._BEM_manager.default_ones,
                    triangle_x=i, triangle_y=j,
                    basis_function_index_x=ii, basis_function_index_y=jj,
                    panels_relation=panels_relation
                )
    
    @ti.func
    def apply_W_dot_boundary(self, vert_boundary, result_vec, cell_type: int, add: int=1):
        assert(vert_boundary.shape[0] == self.num_of_vertices)
        if cell_type == int(CellType.NEUMANN):
            assert(result_vec.shape[0] == self.num_of_Neumanns * self._Q)
        elif cell_type == int(CellType.DIRICHLET):
            assert(result_vec.shape[0] == self.num_of_Dirichlets * self._Q)
        
        multiplier = 1.0
        if not add:
            multiplier = -1.0

        for I in result_vec:
            i = I // self._Q
            if cell_type == int(CellType.DIRICHLET):
                i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(I // self._Q)
            elif cell_type == int(CellType.NEUMANN):
                i = self._BEM_manager.map_local_Neumann_index_to_panel_index(I // self._Q)
            ii = I % self._Q
            for J in range(self.num_of_panels * self._Q):
                j = J // self._Q
                jj = J % self._Q

                panels_relation = self._BEM_manager.get_panels_relation(i, j)
                result_vec[I] += multiplier * self.integrate_on_two_panels(
                    vert_boundary=vert_boundary,
                    triangle_x=i, triangle_y=j,
                    basis_function_index_x=ii, basis_function_index_y=jj,
                    panels_relation=panels_relation
                )
