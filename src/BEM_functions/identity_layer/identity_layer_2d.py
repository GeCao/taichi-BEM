import taichi as ti
import numpy as np

from src.BEM_functions.identity_layer import AbstractIdentityLayer
from src.BEM_functions.utils import CellFluxType, VertAttachType, KernelType, PanelsRelation


class IdentityLayer2d(AbstractIdentityLayer):
    rank = 2

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(IdentityLayer2d, self).__init__(BEM_manager, *args, **kwargs)

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
    def integrate_on_single_panel(
        self,
        rands: ti.types.vector(1, int),
        panel_x: int,
        basis_function_index_x: int,
        basis_function_index_y: int,
    ):
        """
        Get Integration points and weights for a general line

        0---*----*--|1--->x     <- This refers GaussQR = 2, you will get GaussQR=2 points and 2 weights

         ============= TO =============

        0---*----*--|1--->x            
          
                                                x2
                                                /
        0---*----*--|1--->x                    /
                                     ->       /
                                             /
                                          x1/
        
         
         - How to project a unit line to a general one?
         - If we choose (0,)->x1, (1)->x2,
         - x (r1) = (1 - r1) * x1 + r2 * x2

         ==============================
        To sum up, a random number (t, ) as a sample on line, where t \in [0, 1]
        r1,          = ...
        Sampled_point   = (1 - r1) * x1  +  r2 * x2
        Corres_weight   = w * Area
        Jacobian        = r1
        phase functions = 1 - r1, r2

        !!!!!!!!!!!!!!
        However, if these two panels has overlaps, such like common vertices, even the same panel.
        For instance, the same panel:
        Reference: S. A. Sauter and C. Schwab, Boundary Element Methods, Springer Ser. Comput. Math. 39, Springer-Verlag, Berlin, 2011.
        https://link.springer.com/content/pdf/10.1007/978-3-540-68093-2.pdf
        See this algorithm from chapter 5.2.1
        
        Similarly, besides the coincide case we have mentioned above,
        other approaches such like common vertices/edges can be refered by chapter 5.2.2 and 5.2.3
        """
        integrand = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)

        area_x = self._BEM_manager.get_panel_area(panel_x)
        
        # Generate number(xsi,)
        xsi = self._BEM_manager.Gauss_points_1d[rands.x]

        # Scale your weight
        weight_x = self._BEM_manager.Gauss_weights_1d[rands.x]* (area_x * 1.0)
        # Get your final weight
        weight = weight_x

        # Generate number(r1, r2) for panel x
        r1_x = xsi

        # Get your jacobian
        jacobian = 1.0

        phix = self.shape_function(r1_x, -1.0, i=basis_function_index_x)
        phiy = self.shape_function(r1_x, -1.0, i=basis_function_index_y)

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

                        integrand = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
                        for iii in range(self._GaussQR):
                            rands = ti.Vector([iii], ti.i32)

                            integrand += self.integrate_on_single_panel(
                                panel_x=global_i,
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

                        integrand = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
                        for iii in range(self._GaussQR):
                            rands = ti.Vector([iii], ti.i32)
                            
                            integrand += self.integrate_on_single_panel(
                                panel_x=global_i,
                                basis_function_index_x=basis_function_index_x, basis_function_index_y=basis_function_index_y
                            )

                        if local_charge_I >= 0 and local_charge_J >= 0:
                            self._BEM_manager.get_mat_P()[local_charge_I + self._Neumann_offset_i, local_charge_J + self._Dirichlet_offset_j] += mutiplier * integrand
                            self._BEM_manager.get_mat_P()[local_charge_J + self._Dirichlet_offset_i, local_charge_I + self._Neumann_offset_j] += mutiplier * integrand
