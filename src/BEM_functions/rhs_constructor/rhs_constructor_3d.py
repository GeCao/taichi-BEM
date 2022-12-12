import taichi as ti
import numpy as np

from src.BEM_functions.rhs_constructor import AbstractRHSConstructor
from src.BEM_functions.utils import CellFluxType, VertAttachType, KernelType, PanelsRelation, AssembleType


@ti.data_oriented
class RHSConstructor3d(AbstractRHSConstructor):
    rank = 3

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(RHSConstructor3d, self).__init__(BEM_manager, *args, **kwargs)

        self._GaussQR = self._BEM_manager.get_GaussQR()

        self._ti_dtype = self._BEM_manager._ti_dtype
        self._np_dtype = self._BEM_manager._np_dtype
        self._dim = self._BEM_manager._dim
        self._n = self._BEM_manager._n

        self.N_Neumann = self._BEM_manager.get_N_Neumann()
        self._Q_Neumann = self._BEM_manager.get_shape_func_degree_Neumann()
        self.M_Dirichlet = self._BEM_manager.get_M_Dirichlet()
        self._Q_Dirichlet = self._BEM_manager.get_shape_func_degree_Dirichlet()
        self.num_of_vertices = self._BEM_manager.get_num_of_vertices()
        self.num_of_panels = self._BEM_manager.get_num_of_panels()
        self.num_of_panels_Neumann = self._BEM_manager.get_num_of_panels_Neumann()
        self.num_of_panels_Dirichlet = self._BEM_manager.get_num_of_panels_Dirichlet()

        self.analytical_function_Dirichlet = self._BEM_manager.analytical_function_Dirichlet
        self.analytical_function_Neumann = self._BEM_manager.analytical_function_Neumann

        self._Dirichlet_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=())
        self._Neumann_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=())
        if self._Q_Neumann == 0:
            self._Neumann_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_panels,))
        elif self._Q_Neumann == 1:
            self._Neumann_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_vertices,))
        
        if self._Q_Dirichlet == 0:
            self._Dirichlet_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_panels,))
        elif self._Q_Dirichlet == 1:
            self._Dirichlet_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_vertices,))

        self._f_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.N_Neumann + self.M_Dirichlet,))

        self._rhs_vec = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.N_Neumann + self.M_Dirichlet))

    @ti.kernel
    def set_boundaries(self, sqrt_ni: float, sqrt_no: float):
        self._Dirichlet_boundary.fill(0)
        self._Neumann_boundary.fill(0)
        self._f_boundary.fill(0)

        Dirichlet_offset_j = self._BEM_manager.get_Dirichlet_offset_j()
        Neumann_offset_j = self._BEM_manager.get_Neumann_offset_j()
        if ti.static(self._BEM_manager._is_transmission > 0):
            if ti.static(self._Q_Neumann == 0):
                for local_I in range(self.num_of_panels_Neumann):
                    # Solve Neumann, apply Neumann boundary
                    global_i = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_I)
                    # Constant
                    x1 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 0)
                    x2 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 1)
                    x3 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 2)
                    x = (x1 + x2 + x3) / 3.0
                    normal_x = self._BEM_manager.get_panel_normal(global_i)
                    self._Neumann_boundary[global_i] += self.analytical_function_Neumann(x, normal_x, sqrt_ni)
                    self._f_boundary[Neumann_offset_j + local_I] += (
                        self.analytical_function_Neumann(x, normal_x, sqrt_ni) - 0 * self.analytical_function_Neumann(x, normal_x, sqrt_no)
                    )
            elif ti.static(self._Q_Neumann == 1):
                for global_vert_idx_i in range(self.num_of_vertices):
                    # Solve Neumann, apply Neumann boundary
                    if self._BEM_manager.get_vertice_type(global_vert_idx_i) == int(VertAttachType.BOTH_TOBESOLVED) or \
                        self._BEM_manager.get_vertice_type(global_vert_idx_i) == int(VertAttachType.NEUMANN_TOBESOLVED):
                        local_vert_idx_i = self._BEM_manager.map_global_vert_index_to_local_Neumann(global_vert_idx_i)
                        x = self._BEM_manager.get_vertice(global_vert_idx_i)
                        normal_x = self._BEM_manager.get_vert_normal(global_vert_idx_i)
                        self._Neumann_boundary[global_vert_idx_i] += self.analytical_function_Neumann(x, normal_x, sqrt_ni)
                        self._f_boundary[Neumann_offset_j + local_vert_idx_i] += (
                            self.analytical_function_Neumann(x, normal_x, sqrt_ni) - 0 * self.analytical_function_Neumann(x, normal_x, sqrt_no)
                        )
            
            if ti.static(self._Q_Dirichlet == 0):
                for local_I in range(self.num_of_panels_Dirichlet):
                    # Solve Dirichlet, apply Dirichlet boundary
                    global_i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_I)
                    # Constant
                    x1 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 0)
                    x2 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 1)
                    x3 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 2)
                    x = (x1 + x2 + x3) / 3.0
                    normal_x = self._BEM_manager.get_panel_normal(global_i)
                    self._Dirichlet_boundary[global_i] += self.analytical_function_Dirichlet(x, sqrt_ni)
                    self._f_boundary[Dirichlet_offset_j + local_I] += (
                        self.analytical_function_Dirichlet(x, sqrt_ni) - 0 * self.analytical_function_Dirichlet(x, sqrt_no)
                    )
            elif ti.static(self._Q_Dirichlet == 1):
                for global_vert_idx_i in range(self.num_of_vertices):
                    # Solve Dirichlet, apply Dirichlet boundary
                    if self._BEM_manager.get_vertice_type(global_vert_idx_i) == int(VertAttachType.BOTH_TOBESOLVED) or \
                        self._BEM_manager.get_vertice_type(global_vert_idx_i) == int(VertAttachType.DIRICHLET_TOBESOLVED):
                        local_vert_idx_i = self._BEM_manager.map_global_vert_index_to_local_Dirichlet(global_vert_idx_i)
                        x = self._BEM_manager.get_vertice(global_vert_idx_i)
                        normal_x = self._BEM_manager.get_vert_normal(global_vert_idx_i)
                        self._Dirichlet_boundary[global_vert_idx_i] += self.analytical_function_Dirichlet(x, sqrt_ni)
                        self._f_boundary[Dirichlet_offset_j + local_vert_idx_i] += (
                            self.analytical_function_Dirichlet(x, sqrt_ni) - 0 * self.analytical_function_Dirichlet(x, sqrt_no)
                        )
        else:
            if ti.static(self._Q_Neumann == 0):
                for local_I in range(self.num_of_panels_Dirichlet):
                    # Solve Dirichlet
                    global_i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_I)
                    # Constant
                    x1 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 0)
                    x2 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 1)
                    x3 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 2)
                    x = (x1 + x2 + x3) / 3.0
                    normal_x = self._BEM_manager.get_panel_normal(global_i)
                    fx = self.analytical_function_Neumann(x, normal_x)
                    self._Neumann_boundary[global_i] += fx
            elif ti.static(self._Q_Neumann == 1):
                for global_vert_idx_i in range(self.num_of_vertices):
                    if self._BEM_manager.get_vertice_type(global_vert_idx_i) == int(VertAttachType.DIRICHLET_TOBESOLVED):
                        x = self._BEM_manager.get_vertice(global_vert_idx_i)
                        normal_x = self._BEM_manager.get_vert_normal(global_vert_idx_i)
                        fx = self.analytical_function_Neumann(x, normal_x)
                        self._Neumann_boundary[global_vert_idx_i] += fx
        
            if ti.static(self._Q_Dirichlet == 0):
                for local_I in range(self.num_of_panels_Neumann):
                    # Solve Neumann
                    global_i = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_I)
                    # Constant
                    x1 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 0)
                    x2 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 1)
                    x3 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 2)
                    x = (x1 + x2 + x3) / 3.0
                    normal_x = self._BEM_manager.get_panel_normal(global_i)
                    gx = self.analytical_function_Dirichlet(x)
                    self._Dirichlet_boundary[global_i] += gx
            elif ti.static(self._Q_Dirichlet == 1):
                for global_vert_idx_i in range(self.num_of_vertices):
                    if self._BEM_manager.get_vertice_type(global_vert_idx_i) == int(VertAttachType.NEUMANN_TOBESOLVED):
                        x = self._BEM_manager.get_vertice(global_vert_idx_i)
                        normal_x = self._BEM_manager.get_vert_normal(global_vert_idx_i)
                        gx = self.analytical_function_Dirichlet(x)
                        self._Dirichlet_boundary[global_vert_idx_i] += gx
    
    @ti.func
    def get_rhs_vec(self):
        return self._rhs_vec
    
    @ti.func
    def get_Dirichlet_boundary(self, panel_index, basis_function_index):
        result = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
        if basis_function_index == -1:
            # Constant
            result = self._Dirichlet_boundary[panel_index]
        else:
            # Linear
            vert_index = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * panel_index + basis_function_index)
            result = self._Dirichlet_boundary[vert_index]
        
        return result

    @ti.func
    def get_Neumann_boundary(self, panel_index, basis_function_index):
        result = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
        if basis_function_index == -1:
            # Constant
            result = self._Neumann_boundary[panel_index]
        else:
            # Linear
            vert_index = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * panel_index + basis_function_index)
            result = self._Neumann_boundary[vert_index]
        
        return result
    
    @ti.func
    def get_f_boundary(self, local_index):
        return self._f_boundary[local_index]

    def kill(self):
        self.N_Neumann = 0
        self.M_Dirichlet = 0
        self._Q_Neumann = -1
        self._Q_Dirichlet = -1
        self.num_of_vertices = 0
        self.num_of_panels = 0
        self.num_of_panels_Neumann = 0
        self.num_of_panels_Dirichlet = 0

        self._rhs_vec = None
        self._Dirichlet_boundary = None
        self._Neumann_boundary = None
        self._f_boundary = None
    
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
    def integrate_on_single_triangle(
        self,
        triangle_x: int,
        basis_function_index_x: int,
        basis_function_index_y: int,
    ):
        """Get Integration on a single triangle, where
        int_{Tau_x} (func) dx
            = (2 * area_x) * int_{unit_triangle} (func) (Jacobian) dr1 dr2
        
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
        integrand = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)

        area_x = self._BEM_manager.get_panel_area(triangle_x)

        GaussQR2 = self._GaussQR * self._GaussQR
        
        for iii in range(GaussQR2):
            # Generate number(r1, r2)
            r1_x = self._BEM_manager.Gauss_points_1d[iii // self._GaussQR]
            r2_x = self._BEM_manager.Gauss_points_1d[iii % self._GaussQR] * r1_x

            # Scale your weight
            weight_x = self._BEM_manager.Gauss_weights_1d[iii // self._GaussQR] * self._BEM_manager.Gauss_weights_1d[iii % self._GaussQR] * (area_x * 2.0)
            # Get your final weight
            weight = weight_x

            jacobian = r1_x

            phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
            phiy = self.shape_function(r1_x, r2_x, i=basis_function_index_y)

            integrand.x += (
                phix * phiy
            ) * weight * jacobian
        
        return integrand
    
    @ti.kernel
    def multiply_identity_to_vector(self, multiplier: float):
        basis_func_num_Neumann = self._BEM_manager.get_num_of_basis_functions_from_Q(self._Q_Neumann)
        basis_func_num_Dirichlet = self._BEM_manager.get_num_of_basis_functions_from_Q(self._Q_Dirichlet)
        Neumann_offset_i = self._BEM_manager.get_Neumann_offset_i()
        Dirichlet_offset_i = self._BEM_manager.get_Dirichlet_offset_i()

        if ti.static(self.N_Neumann > 0):
            # += 0.5I * g
            for local_I in range(self.num_of_panels_Neumann):
                global_i = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_I)
                for ii in range(basis_func_num_Neumann):
                    for jj in range(basis_func_num_Dirichlet):
                        basis_function_index_x = self._BEM_manager.get_basis_function_index(self._Q_Neumann, ii)
                        basis_function_index_y = self._BEM_manager.get_basis_function_index(self._Q_Dirichlet, jj)

                        local_charge_I = self._BEM_manager.proj_from_local_panel_index_to_local_charge_index(
                            Q_=self._Q_Neumann,
                            local_panel_index=local_I,
                            basis_func_index=basis_function_index_x,
                            panel_type=int(CellFluxType.NEUMANN_TOBESOLVED)
                        )

                        global_charge_j = self._BEM_manager.proj_from_local_panel_index_to_global_charge_index(
                            Q_=self._Q_Dirichlet,
                            local_panel_index=local_I,
                            basis_func_index=basis_function_index_y,
                            panel_type=int(CellFluxType.NEUMANN_TOBESOLVED)
                        )
                        
                        integrand = self.integrate_on_single_triangle(
                            triangle_x=global_i,
                            basis_function_index_x=basis_function_index_x, basis_function_index_y=basis_function_index_y
                        )

                        gx = self._Dirichlet_boundary[global_charge_j]

                        if local_charge_I >= 0:
                            if ti.static(self._n == 1):
                                self._rhs_vec[local_charge_I + Neumann_offset_i] += multiplier * integrand * gx
                            elif ti.static(self._n == 2):
                                self._rhs_vec[local_charge_I + Neumann_offset_i] += multiplier * ti.math.cmul(integrand, gx)
            
        if ti.static(self.M_Dirichlet > 0):
            # += 0.5I * f
            for local_I in range(self.num_of_panels_Dirichlet):
                global_i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_I)
                for ii in range(basis_func_num_Dirichlet):
                    for jj in range(basis_func_num_Neumann):
                        basis_function_index_x = self._BEM_manager.get_basis_function_index(self._Q_Dirichlet, ii)
                        basis_function_index_y = self._BEM_manager.get_basis_function_index(self._Q_Neumann, jj)

                        local_charge_I = self._BEM_manager.proj_from_local_panel_index_to_local_charge_index(
                            Q_=self._Q_Dirichlet,
                            local_panel_index=local_I,
                            basis_func_index=basis_function_index_x,
                            panel_type=int(CellFluxType.DIRICHLET_TOBESOLVED)
                        )

                        global_charge_j = self._BEM_manager.proj_from_local_panel_index_to_global_charge_index(
                            Q_=self._Q_Neumann,
                            local_panel_index=local_I,
                            basis_func_index=basis_function_index_y,
                            panel_type=int(CellFluxType.DIRICHLET_TOBESOLVED)
                        )

                        integrand = self.integrate_on_single_triangle(
                            triangle_x=global_i,
                            basis_function_index_x=basis_function_index_x, basis_function_index_y=basis_function_index_y
                        )

                        fx = self._Neumann_boundary[global_charge_j]

                        if local_charge_I >= 0:
                            if ti.static(self._n == 1):
                                self._rhs_vec[local_charge_I + Dirichlet_offset_i] += multiplier * integrand * fx
                            elif ti.static(self._n == 2):
                                self._rhs_vec[local_charge_I + Dirichlet_offset_i] += multiplier * ti.math.cmul(integrand, fx)
    
    def multiply_M_to_vector(self, k: float, sqrt_n: float, multiplier: float):
        if self.N_Neumann > 0:
            # += K * g
            self._BEM_manager.double_layer.apply_K_dot_vert_boundary(k=k, sqrt_n=sqrt_n, multiplier=multiplier)
            # -= V * f
            self._BEM_manager.single_layer.apply_V_dot_panel_boundary(k=k, sqrt_n=sqrt_n, multiplier=-multiplier)

        if self.M_Dirichlet > 0:
            # -= K' * f
            self._BEM_manager.adj_double_layer.apply_K_dot_panel_boundary(k=k, sqrt_n=sqrt_n, multiplier=-multiplier)
            # -= W * g
            self._BEM_manager.hypersingular_layer.apply_W_dot_vert_boundary(k=k, sqrt_n=sqrt_n, multiplier=-multiplier)
    
    def forward(self, assemble_type: int, k: float, sqrt_n: float):
        self._rhs_vec.fill(0)

        if ti.static(self._BEM_manager._is_transmission <= 0):
            self.multiply_identity_to_vector(multiplier=0.5)

            if assemble_type == int(AssembleType.ADD_P_MINUS):
                self.multiply_M_to_vector(k=k, sqrt_n=sqrt_n, multiplier=-1.0)
            elif assemble_type == int(AssembleType.ADD_P_PLUS):
                self.multiply_M_to_vector(k=k, sqrt_n=sqrt_n, multiplier=1.0)
