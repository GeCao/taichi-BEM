import taichi as ti
import numpy as np

from src.BEM_functions.single_layer import AbstractSingleLayer
from src.managers.mesh_manager import CellFluxType, VertAttachType, PanelsRelation


class SingleLayer3d(AbstractSingleLayer):
    rank = 3

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(SingleLayer3d, self).__init__(BEM_manager, *args, **kwargs)

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
        assert(self.num_of_Neumanns + self.num_of_Dirichlets > 0)
        self._Vmat = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=())
        if self.num_of_Dirichlets > 0:
            self._Vmat = ti.Vector.field(
                self._n,
                dtype=self._ti_dtype,
                shape=(self.num_of_Dirichlets, self.num_of_Dirichlets)
            )
    
    @ti.func
    def get_V_mat(self):
        return self._Vmat

    def kill(self):
        self.num_of_Dirichlets = 0
        self.num_of_Neumanns = 0
        self._Vmat = None

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
        return self._BEM_manager.shape_function(r1, r2, i)
    
    @ti.func
    def integrate_on_two_panels(
        self,
        panel_boundary,
        triangle_x: int,
        triangle_y: int,
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
        """
        integrand = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)

        x1 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_x + 0)
        x2 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_x + 1)
        x3 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_x + 2)
        area_x = self._BEM_manager.get_panel_area(triangle_x)
        normal_x = self._BEM_manager.get_panel_normal(triangle_x)

        y1 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_y + 0)
        y2 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_y + 1)
        y3 = self._BEM_manager.get_vertice_from_flat_panel_index(3 * triangle_y + 2)
        area_y = self._BEM_manager.get_panel_area(triangle_y)
        normal_y = self._BEM_manager.get_panel_normal(triangle_y)

        inner_val_y = panel_boundary[triangle_y]

        GaussQR2 = self._GaussQR * self._GaussQR
        GaussQR4 = GaussQR2 * GaussQR2
        
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

                integrand += (
                    self.G(x, y) * inner_val_y
                ) * weight * jacobian
            elif panels_relation == int(PanelsRelation.COINCIDE):
                # Get your jacobian
                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2

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

                    # D1, D3, D5
                    integrand += (
                        self.G(x, y) * inner_val_y
                    ) * weight * jacobian

                    r1_y, r2_y = xz[0], xz[1]
                    r1_x, r2_x = xz[0] - xz[2], xz[1] - xz[3]

                    x = self.interplate_from_unit_triangle_to_general(
                        r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                    )
                    y = self.interplate_from_unit_triangle_to_general(
                        r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                    )

                    # D2, D4, D6
                    integrand += (
                        self.G(x, y) * inner_val_y
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
                
                jacobian = xsi * xsi * xsi * eta2
                integrand += (
                    self.G(x, y) * inner_val_y
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
                
                jacobian = xsi * xsi * xsi * eta2
                integrand += (
                    self.G(x, y) * inner_val_y
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
                
                jacobian = xsi * xsi * xsi * eta1 * eta1
                integrand += (
                    self.G(x, y) * inner_val_y
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
                
                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
                integrand += (
                    self.G(x, y) * inner_val_y
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
                
                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
                integrand += (
                    self.G(x, y) * inner_val_y
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

                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
                integrand += (
                    self.G(x, y) * inner_val_y
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

                jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
                integrand += (
                    self.G(x, y) * inner_val_y
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
        if ti.static(self.num_of_Dirichlets > 0):
            self._Vmat.fill(0)

            for local_I, local_J in self._Vmat:
                global_i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_I)
                global_j = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_J)

                # Construct a local matrix
                panels_relation = self._BEM_manager.get_panels_relation(global_i, global_j)
                self._Vmat[local_I, local_J] += self.integrate_on_two_panels(
                    panel_boundary=self._BEM_manager.default_panel_ones,
                    triangle_x=global_i, triangle_y=global_j,
                    panels_relation=panels_relation
                )
    
    @ti.func
    def apply_V_dot_panel_boundary(self, panel_boundary, result_vec, add: int=1):
        assert(panel_boundary.shape[0] == self.num_of_panels)
        assert(result_vec.shape[0] == self.num_of_Dirichlets)
        
        multiplier = 1.0
        if not add:
            multiplier = -1.0

        num_of_Neumann_panels = self.num_of_panels - self.num_of_Dirichlets
        for local_I in result_vec:
            for local_J in range(num_of_Neumann_panels):
                global_i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_I)
                global_j = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_J)
                panels_relation = self._BEM_manager.get_panels_relation(global_i, global_j)
                result_vec[local_I] += multiplier * self.integrate_on_two_panels(
                    panel_boundary=panel_boundary,
                    triangle_x=global_i, triangle_y=global_j,
                    panels_relation=panels_relation
                )
