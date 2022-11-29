import taichi as ti
import numpy as np

from src.BEM_functions.hypersingular_layer import AbstractHypersingularLayer
from src.BEM_functions.utils import CellFluxType, VertAttachType, KernelType, PanelsRelation


class HypersingularLayer3d(AbstractHypersingularLayer):
    rank = 3

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(HypersingularLayer3d, self).__init__(BEM_manager, *args, **kwargs)

        self._GaussQR = self._BEM_manager._GaussQR
        self._Q = self._BEM_manager._Q  # Number of local shape functions
        self._dim = self._BEM_manager._dim

        self._ti_dtype = self._BEM_manager._ti_dtype
        self._np_dtype = self._BEM_manager._np_dtype
        self._kernel_type = self._BEM_manager._kernel_type
        self._k = self._BEM_manager._k
        self._sqrt_n = 1.0
        self._n = self._BEM_manager._n

        self.num_of_Dirichlets = self._BEM_manager.get_num_of_Dirichlets()
        self.num_of_Neumanns = self._BEM_manager.get_num_of_Neumanns()
        self.num_of_vertices = self._BEM_manager.get_num_of_vertices()
        self.num_of_panels = self._BEM_manager.get_num_of_panels()
        self._Wmat = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=())
        assert(self.num_of_Dirichlets + self.num_of_Neumanns > 0)
        if self.num_of_Neumanns > 0:
            self._Wmat = ti.Vector.field(
                self._n,
                dtype=self._ti_dtype,
                shape=(self.num_of_Neumanns, self.num_of_Neumanns)
            )
    
    @ti.func
    def get_W_mat(self):
        return self._Wmat
    
    def kill(self):
        self.num_of_Dirichlets = 0
        self.num_of_Neumanns = 0
        self._Wmat = None

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
        grad_u = (
            u1 * ti.math.cross(normal, x3 - x2) +
            u2 * ti.math.cross(normal, x1 - x3) +
            u3 * ti.math.cross(normal, x2 - x1)
        ) / (2.0 * area)  #  grad_p times face area

        return grad_u
    
    @ti.func
    def shape_function(self, r1, r2, i: int):
        return self._BEM_manager.shape_function(r1, r2, i)
    
    @ti.func
    def integrate_on_two_panels(
        self,
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
        To sum up, a random number (t, s) as a sample on triangle, where t,s \in [0, 1]
        r1, r2          = ...
        Sampled_point   = (1 - r1) * x1  +  (r1 - r2) * x2  +  r2 * x3
        Corres_weight   = w * 2 * Area
        Jacobian        = r1
        phase functions = 1 - r1, r1 - r2, r2

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
        
        Similarly, besides the coincide case we have mentioned above,
        other approaches such like common vertices/edges can be refered by chapter 5.2.2 and 5.2.2
        """
        integrand = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)

        x1 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * triangle_x + 0)
        x2 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * triangle_x + 1)
        x3 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * triangle_x + 2)
        area_x = self._BEM_manager.get_panel_area(triangle_x)
        normal_x = self._BEM_manager.get_panel_normal(triangle_x)
        phi1_x = 1.0 * (basis_function_index_x == 0)
        phi2_x = 1.0 * (basis_function_index_x == 1)
        phi3_x = 1.0 * (basis_function_index_x == 2)

        y1 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * triangle_y + 0)
        y2 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * triangle_y + 1)
        y3 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * triangle_y + 2)
        area_y = self._BEM_manager.get_panel_area(triangle_y)
        normal_y = self._BEM_manager.get_panel_normal(triangle_y)
        phi1_y = 1.0 * (basis_function_index_y == 0)
        phi2_y = 1.0 * (basis_function_index_y == 1)
        phi3_y = 1.0 * (basis_function_index_y == 2)

        surface_grad_i_x = self.surface_grad(x1, x2, x3, phi1_x, phi2_x, phi3_x, normal_x, area_x)
        curl_i_x = ti.math.cross(normal_x, surface_grad_i_x)

        surface_grad_i_y = self.surface_grad(y1, y2, y3, phi1_y, phi2_y, phi3_y, normal_y, area_y)
        curl_i_y = ti.math.cross(normal_y, surface_grad_i_y)

        curl_phix_dot_curl_phiy = curl_i_x.dot(curl_i_y)

        GaussQR2 = self._GaussQR * self._GaussQR
        GaussQR4 = GaussQR2 * GaussQR2
        k2 = self._k * self._k
        
        for gauss_number in range(GaussQR4):
            iii = gauss_number // GaussQR2
            jjj = gauss_number % GaussQR2

            # Generate number(xsi, eta1, eta2, eta3)
            xsi = self._BEM_manager.Gauss_points_1d[iii // self._GaussQR]
            eta1 = self._BEM_manager.Gauss_points_1d[iii % self._GaussQR]
            eta2 = self._BEM_manager.Gauss_points_1d[jjj // self._GaussQR]
            eta3 = self._BEM_manager.Gauss_points_1d[jjj % self._GaussQR]

            # Scale your weight
            weight_x = self._BEM_manager.Gauss_weights_1d[iii // self._GaussQR] * self._BEM_manager.Gauss_weights_1d[iii % self._GaussQR] * (area_x * 2.0)
            weight_y = self._BEM_manager.Gauss_weights_1d[jjj // self._GaussQR] * self._BEM_manager.Gauss_weights_1d[jjj % self._GaussQR] * (area_y * 2.0)
            # Get your final weight
            weight = weight_x * weight_y

            if panels_relation == int(PanelsRelation.SEPARATE):
                # Generate number(r1, r2) for panel x
                r1_x = xsi
                r2_x = eta1 * r1_x
                # Generate number(r1, r2) for panel y
                r1_y = eta2
                r2_y = eta3 * r1_y

                # Get your jacobian
                jacobian = r1_x * r1_y

                x = self.interplate_from_unit_triangle_to_general(
                    r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                )
                y = self.interplate_from_unit_triangle_to_general(
                    r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                )
                phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

                integrand += (
                    self.grad2_G_xy(x, y, curl_phix_dot_curl_phiy, self._sqrt_n)
                ) * weight * jacobian
                integrand += k2 * (
                    -self.G(x, y, self._sqrt_n) * phix * phiy * ti.math.dot(normal_x, normal_y)
                ) * weight * jacobian
            elif panels_relation == int(PanelsRelation.COINCIDE):
                # Get your jacobian
                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2

                # This algorithm includes 6 regions D1 ~ D6
                # By symmetic of kernel, we can simply compress it into 3 regions
                w = ti.Vector([xsi, xsi * eta1, xsi * eta1 * eta2, xsi * eta1 * eta2 * eta3])
                for iiii in range(self._BEM_manager.m_mats_coincide.shape[0]):
                    xz = self._BEM_manager.m_mats_coincide[iiii] @ w  # On unit triangle
                    
                    r1_x, r2_x = xz[0], xz[1]
                    r1_y, r2_y = xz[0] - xz[2], xz[1] - xz[3]

                    x = self.interplate_from_unit_triangle_to_general(
                        r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                    )
                    y = self.interplate_from_unit_triangle_to_general(
                        r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                    )
                    phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                    phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

                    # D1, D3, D5
                    integrand += (
                        self.grad2_G_xy(x, y, curl_phix_dot_curl_phiy, self._sqrt_n)
                    ) * weight * jacobian
                    integrand += k2 * (
                        -self.G(x, y, self._sqrt_n) * phix * phiy * ti.math.dot(normal_x, normal_y)
                    ) * weight * jacobian

                    r1_y, r2_y = xz[0], xz[1]
                    r1_x, r2_x = xz[0] - xz[2], xz[1] - xz[3]

                    x = self.interplate_from_unit_triangle_to_general(
                        r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                    )
                    y = self.interplate_from_unit_triangle_to_general(
                        r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                    )
                    phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                    phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

                    # D2, D4, D6
                    integrand += (
                        self.grad2_G_xy(x, y, curl_phix_dot_curl_phiy, self._sqrt_n)
                    ) * weight * jacobian
                    integrand += k2 * (
                        -self.G(x, y, self._sqrt_n) * phix * phiy * ti.math.dot(normal_x, normal_y)
                    ) * weight * jacobian
            elif panels_relation == int(PanelsRelation.COMMON_VERTEX):
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
                phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

                jacobian = xsi * xsi * xsi * eta2
                integrand += (
                    self.grad2_G_xy(x, y, curl_phix_dot_curl_phiy, self._sqrt_n)
                ) * weight * jacobian
                integrand += k2 * (
                    -self.G(x, y, self._sqrt_n) * phix * phiy * ti.math.dot(normal_x, normal_y)
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
                phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

                jacobian = xsi * xsi * xsi * eta2
                integrand += (
                    self.grad2_G_xy(x, y, curl_phix_dot_curl_phiy, self._sqrt_n)
                ) * weight * jacobian
                integrand += k2 * (
                    -self.G(x, y, self._sqrt_n) * phix * phiy * ti.math.dot(normal_x, normal_y)
                ) * weight * jacobian
            elif panels_relation == int(PanelsRelation.COMMON_EDGE):
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
                phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

                jacobian = xsi * xsi * xsi * eta1 * eta1
                integrand += (
                    self.grad2_G_xy(x, y, curl_phix_dot_curl_phiy, self._sqrt_n)
                ) * weight * jacobian
                integrand += k2 * (
                    -self.G(x, y, self._sqrt_n) * phix * phiy * ti.math.dot(normal_x, normal_y)
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
                phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
                integrand += (
                    self.grad2_G_xy(x, y, curl_phix_dot_curl_phiy, self._sqrt_n)
                ) * weight * jacobian
                integrand += k2 * (
                    -self.G(x, y, self._sqrt_n) * phix * phiy * ti.math.dot(normal_x, normal_y)
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
                phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
                integrand += (
                    self.grad2_G_xy(x, y, curl_phix_dot_curl_phiy, self._sqrt_n)
                ) * weight * jacobian
                integrand += k2 * (
                    -self.G(x, y, self._sqrt_n) * phix * phiy * ti.math.dot(normal_x, normal_y)
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
                phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
                integrand += (
                    self.grad2_G_xy(x, y, curl_phix_dot_curl_phiy, self._sqrt_n)
                ) * weight * jacobian
                integrand += k2 * (
                    -self.G(x, y, self._sqrt_n) * phix * phiy * ti.math.dot(normal_x, normal_y)
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
                phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)
                
                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
                integrand += (
                    self.grad2_G_xy(x, y, curl_phix_dot_curl_phiy, self._sqrt_n)
                ) * weight * jacobian
                integrand += k2 * (
                    -self.G(x, y, self._sqrt_n) * phix * phiy * ti.math.dot(normal_x, normal_y)
                ) * weight * jacobian
        
        return integrand
    
    @ti.kernel
    def forward(self):
        """
        Compute BIO matix W_mat
        Please note other than other three BIOs, this BIO has a negtive sign
        """
        if ti.static(self.num_of_Neumanns > 0):
            self._Wmat.fill(0)

            num_of_Neumann_panels = self.num_of_panels - self.num_of_Dirichlets
            range_ = (num_of_Neumann_panels * self._dim) * (num_of_Neumann_panels * self._dim)
            for iii in range(range_):
                i = iii // (num_of_Neumann_panels * self._dim)
                j = iii % (num_of_Neumann_panels * self._dim)
                local_I = i // self._dim
                local_J = j // self._dim
                ii = i % self._dim
                jj = j % self._dim

                global_i = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_I)
                global_j = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_J)

                panels_relation = self._BEM_manager.get_panels_relation(global_i, global_j)

                global_vert_idx_i = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * global_i + ii)
                local_vert_idx_i = self._BEM_manager.map_global_vert_index_to_local_Neumann(global_vert_idx_i)
                global_vert_idx_j = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * global_j + jj)
                local_vert_idx_j = self._BEM_manager.map_global_vert_index_to_local_Neumann(global_vert_idx_j)
                self._Wmat[local_vert_idx_i, local_vert_idx_j] += self.integrate_on_two_panels(
                    triangle_x=global_i, triangle_y=global_j,
                    basis_function_index_x=ii, basis_function_index_y=jj,
                    panels_relation=panels_relation
                )
    
    @ti.func
    def apply_W_dot_vert_boundary(self, vert_boundary, result_vec, add: int=1):
        """
        If you are applying Neumann boundary on panels,
        You need to solve a linear system equations to get Dirichlet vertices,
        Usually, a linear system equations requires a rhs, which

        rhs_{Neumann part} = Mf / 2 - K'f + Wg

        In this function, a [Wg] will be computed,
        where [g] is the input argument [vert_boundary] where an extended Dirichlet boundary is applied on vertices
        and [W] is our own BIO matrix [self._Wmat]
        """
        assert(vert_boundary.shape[0] == self.num_of_vertices)
        assert(result_vec.shape[0] == self.num_of_Neumanns)
        
        multiplier = 2.0 * (add > 0) - 1.0
        
        num_of_Neumann_panels = self.num_of_panels - self.num_of_Dirichlets
        for local_I in range(num_of_Neumann_panels):
            for local_J in range(self.num_of_Dirichlets):
                for ii in range(self._dim):
                    global_i = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_I)
                    global_vert_idx = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * global_i + ii)
                    local_vert_idx = self._BEM_manager.map_global_vert_index_to_local_Neumann(global_vert_idx)
                    global_j = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_J)

                    g1 = vert_boundary[self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * global_j + 0)]
                    g2 = vert_boundary[self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * global_j + 1)]
                    g3 = vert_boundary[self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * global_j + 2)]
                    gy = (g1 + g2 + g3) / 3.0

                    panels_relation = self._BEM_manager.get_panels_relation(global_i, global_j)
                    result_vec[local_vert_idx] += multiplier * self.integrate_on_two_panels(
                        triangle_x=global_i, triangle_y=global_j,
                        basis_function_index_x=ii, basis_function_index_y=-1,
                        panels_relation=panels_relation
                    ) * gy
