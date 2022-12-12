import taichi as ti
import numpy as np

from src.BEM_functions.identity_layer import AbstractIdentityLayer
from src.BEM_functions.utils import CellFluxType, VertAttachType, KernelType, PanelsRelation


class IdentityLayer3d(AbstractIdentityLayer):
    rank = 3

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(IdentityLayer3d, self).__init__(BEM_manager, *args, **kwargs)

        self._GaussQR = self._BEM_manager.get_GaussQR()
        self._dim = self._BEM_manager._dim

        self._ti_dtype = self._BEM_manager._ti_dtype
        self._np_dtype = self._BEM_manager._np_dtype
        self._n = self._BEM_manager._n

        self.N_Neumann = self._BEM_manager.get_N_Neumann()
        self._Q_Neumann = self._BEM_manager.get_shape_func_degree_Neumann()
        self.M_Dirichlet = self._BEM_manager.get_M_Dirichlet()
        self._Q_Dirichlet = self._BEM_manager.get_shape_func_degree_Dirichlet()
        self.num_of_vertices = self._BEM_manager.get_num_of_vertices()
        self.num_of_panels = self._BEM_manager.get_num_of_panels()
        self.num_of_panels_Neumann = self._BEM_manager.get_num_of_panels_Neumann()
        self.num_of_panels_Dirichlet = self._BEM_manager.get_num_of_panels_Dirichlet()

        self._Dirichlet_offset_i = self._BEM_manager._Dirichlet_offset_i
        self._Neumann_offset_i = self._BEM_manager._Neumann_offset_i
        self._Dirichlet_offset_j = self._BEM_manager._Dirichlet_offset_j
        self._Neumann_offset_j = self._BEM_manager._Neumann_offset_j
    
    def kill(self):
        self.N_Neumann = 0
        self.M_Dirichlet = 0
        self._Q_Neumann = -1
        self._Q_Dirichlet = -1
        self.num_of_vertices = 0
        self.num_of_panels = 0
        self.num_of_panels_Neumann = 0
        self.num_of_panels_Dirichlet = 0

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
    def integrate_on_single_panel(
        self,
        triangle_x: int,
        basis_function_index_x: int,
        basis_function_index_y: int,
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
        panel_type = self._BEM_manager.get_panel_type(triangle_x)

        GaussQR2 = self._GaussQR * self._GaussQR
        
        for gauss_number in range(GaussQR2):
            iii = gauss_number

            # Generate number(xsi, eta1, eta2, eta3)
            xsi = self._BEM_manager.Gauss_points_1d[iii // self._GaussQR]
            eta1 = self._BEM_manager.Gauss_points_1d[iii % self._GaussQR]

            # Scale your weight
            weight_x = self._BEM_manager.Gauss_weights_1d[iii // self._GaussQR] * self._BEM_manager.Gauss_weights_1d[iii % self._GaussQR] * (area_x * 2.0)
            # Get your final weight
            weight = weight_x

            # Generate number(r1, r2) for panel x
            r1_x = xsi
            r2_x = eta1 * r1_x

            # Get your jacobian
            jacobian = r1_x

            phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
            phiy = self.shape_function(r1_x, r2_x, i=basis_function_index_y)

            integrand.x += (
                phix * phiy
            ) * weight * jacobian
        
        return integrand
    
    @ti.kernel
    def matA_add_global_Identity_matrix(self, mutiplier: float):
        """
        Compute BIO matix W_mat
        Please note other than other three BIOs, this BIO has a negtive sign
        """
        if ti.static(self._BEM_manager._is_transmission > 0):
            basis_func_num_Neumann = self._BEM_manager.get_num_of_basis_functions_from_Q(self._Q_Neumann)
            basis_func_num_Dirichlet = self._BEM_manager.get_num_of_basis_functions_from_Q(self._Q_Dirichlet)
            for local_I in range(self.num_of_panels):
                global_i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_I)
                for ii in range(basis_func_num_Neumann):
                    for jj in range(basis_func_num_Dirichlet):
                        basis_function_index_x = self._BEM_manager.get_basis_function_index(self._Q_Neumann, ii)
                        basis_function_index_y = self._BEM_manager.get_basis_function_index(self._Q_Dirichlet, jj)

                        # Different from other cases, here the true panel type is: BOTH_TOBESOVLED,
                        # Which implies, you can not get_panel_type automatically, but identify them locally
                        local_charge_I = self._BEM_manager.proj_from_local_panel_index_to_local_charge_index(
                            Q_=self._Q_Neumann,
                            local_panel_index=local_I,
                            basis_func_index=basis_function_index_x,
                            panel_type=int(CellFluxType.NEUMANN_TOBESOLVED)
                        )
                        local_charge_J = self._BEM_manager.proj_from_local_panel_index_to_local_charge_index(
                            Q_=self._Q_Dirichlet,
                            local_panel_index=local_I,
                            basis_func_index=basis_function_index_y,
                            panel_type=int(CellFluxType.DIRICHLET_TOBESOLVED)
                        )

                        integrand = self.integrate_on_single_panel(
                            triangle_x=global_i,
                            basis_function_index_x=basis_function_index_x, basis_function_index_y=basis_function_index_y
                        )

                        if local_charge_I >= 0 and local_charge_J >= 0:
                            self._BEM_manager.get_mat_A()[local_charge_I + self._Neumann_offset_i, local_charge_J + self._Dirichlet_offset_j] += mutiplier * integrand
                            self._BEM_manager.get_mat_A()[local_charge_J + self._Dirichlet_offset_i, local_charge_I + self._Neumann_offset_j] += mutiplier * integrand

    
    @ti.kernel
    def matP_add_global_Identity_matrix(self, mutiplier: float):
        """
        Compute BIO matix W_mat
        Please note other than other three BIOs, this BIO has a negtive sign
        """
        if ti.static(self._BEM_manager._is_transmission > 0):
            basis_func_num_Neumann = self._BEM_manager.get_num_of_basis_functions_from_Q(self._Q_Neumann)
            basis_func_num_Dirichlet = self._BEM_manager.get_num_of_basis_functions_from_Q(self._Q_Dirichlet)
            for local_I in range(self.num_of_panels):
                global_i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_I)
                for ii in range(basis_func_num_Neumann):
                    for jj in range(basis_func_num_Dirichlet):
                        basis_function_index_x = self._BEM_manager.get_basis_function_index(self._Q_Neumann, ii)
                        basis_function_index_y = self._BEM_manager.get_basis_function_index(self._Q_Dirichlet, jj)

                        # Different from other cases, here the true panel type is: BOTH_TOBESOVLED,
                        # Which implies, you can not get_panel_type automatically, but identify them locally
                        local_charge_I = self._BEM_manager.proj_from_local_panel_index_to_local_charge_index(
                            Q_=self._Q_Neumann,
                            local_panel_index=local_I,
                            basis_func_index=basis_function_index_x,
                            panel_type=int(CellFluxType.NEUMANN_TOBESOLVED)
                        )
                        local_charge_J = self._BEM_manager.proj_from_local_panel_index_to_local_charge_index(
                            Q_=self._Q_Dirichlet,
                            local_panel_index=local_I,
                            basis_func_index=basis_function_index_y,
                            panel_type=int(CellFluxType.DIRICHLET_TOBESOLVED)
                        )

                        integrand = self.integrate_on_single_panel(
                            triangle_x=global_i,
                            basis_function_index_x=basis_function_index_x, basis_function_index_y=basis_function_index_y
                        )

                        if local_charge_I >= 0 and local_charge_J >= 0:
                            self._BEM_manager.get_mat_P()[local_charge_I + self._Neumann_offset_i, local_charge_J + self._Dirichlet_offset_j] += mutiplier * integrand
                            self._BEM_manager.get_mat_P()[local_charge_J + self._Dirichlet_offset_i, local_charge_I + self._Neumann_offset_j] += mutiplier * integrand
