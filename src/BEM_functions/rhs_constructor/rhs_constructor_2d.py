import taichi as ti
import numpy as np

from src.BEM_functions.rhs_constructor import AbstractRHSConstructor
from src.BEM_functions.utils import CellFluxType, VertAttachType, KernelType, PanelsRelation, AssembleType, ScopeType


@ti.data_oriented
class RHSConstructor2d(AbstractRHSConstructor):
    rank = 2

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(RHSConstructor2d, self).__init__(BEM_manager, *args, **kwargs)

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
    def set_boundaries(self, scope_type: int, sqrt_ni: float, sqrt_no: float):
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
                    x = ti.Vector([0.0 for j in range(self._dim)], self._ti_dtype)
                    for ii in range(self._dim):
                        x += self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + ii) / float(self._dim)
                    
                    normal_x_int = self._BEM_manager.get_panel_normal(global_i, scope_type=int(ScopeType.INTERIOR))
                    normal_x_ext = -normal_x_int
                    
                    self._Neumann_boundary[global_i] += self.analytical_function_Neumann(x, normal_x, sqrt_ni)
                    self._f_boundary[Neumann_offset_j + local_I] += (
                        self.analytical_function_Neumann(x, normal_x_int, sqrt_ni) - self.analytical_function_Neumann(x, normal_x_ext, sqrt_no)
                    )
            elif ti.static(self._Q_Neumann == 1):
                for global_vert_idx_i in range(self.num_of_vertices):
                    # Solve Neumann, apply Neumann boundary
                    if self._BEM_manager.get_vertice_type(global_vert_idx_i) == int(VertAttachType.BOTH_TOBESOLVED) or \
                        self._BEM_manager.get_vertice_type(global_vert_idx_i) == int(VertAttachType.NEUMANN_TOBESOLVED):
                        local_vert_idx_i = self._BEM_manager.map_global_vert_index_to_local_Neumann(global_vert_idx_i)
                        x = self._BEM_manager.get_vertice(global_vert_idx_i)
                        normal_x_int = self._BEM_manager.get_vert_normal(global_vert_idx_i, scope_type=int(ScopeType.INTERIOR))
                        normal_x_ext = -normal_x_int

                        self._Neumann_boundary[global_vert_idx_i] += self.analytical_function_Neumann(x, normal_x, sqrt_ni)
                        self._f_boundary[Neumann_offset_j + local_vert_idx_i] += (
                            self.analytical_function_Neumann(x, normal_x_int, sqrt_ni) - self.analytical_function_Neumann(x, normal_x_ext, sqrt_no)
                        )
            
            if ti.static(self._Q_Dirichlet == 0):
                for local_I in range(self.num_of_panels_Dirichlet):
                    # Solve Dirichlet, apply Dirichlet boundary
                    global_i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_I)
                    # Constant
                    x = ti.Vector([0.0 for j in range(self._dim)], self._ti_dtype)
                    for ii in range(self._dim):
                        x += self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + ii) / float(self._dim)
                    
                    self._Dirichlet_boundary[global_i] += self.analytical_function_Dirichlet(x, sqrt_ni)
                    self._f_boundary[Dirichlet_offset_j + local_I] += (
                        self.analytical_function_Dirichlet(x, sqrt_ni) - self.analytical_function_Dirichlet(x, sqrt_no)
                    )
            elif ti.static(self._Q_Dirichlet == 1):
                for global_vert_idx_i in range(self.num_of_vertices):
                    # Solve Dirichlet, apply Dirichlet boundary
                    if self._BEM_manager.get_vertice_type(global_vert_idx_i) == int(VertAttachType.BOTH_TOBESOLVED) or \
                        self._BEM_manager.get_vertice_type(global_vert_idx_i) == int(VertAttachType.DIRICHLET_TOBESOLVED):
                        local_vert_idx_i = self._BEM_manager.map_global_vert_index_to_local_Dirichlet(global_vert_idx_i)
                        x = self._BEM_manager.get_vertice(global_vert_idx_i)
                        self._Dirichlet_boundary[global_vert_idx_i] += self.analytical_function_Dirichlet(x, sqrt_ni)
                        self._f_boundary[Dirichlet_offset_j + local_vert_idx_i] += (
                            self.analytical_function_Dirichlet(x, sqrt_ni) - self.analytical_function_Dirichlet(x, sqrt_no)
                        )
        else:
            if ti.static(self._Q_Neumann == 0):
                for local_I in range(self.num_of_panels_Dirichlet):
                    # Solve Dirichlet
                    global_i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_I)
                    # Constant
                    x = ti.Vector([0.0 for j in range(self._dim)], self._ti_dtype)
                    for ii in range(self._dim):
                        x += self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + ii) / float(self._dim)
                    
                    normal_x = self._BEM_manager.get_panel_normal(global_i, scope_type=scope_type)
                    fx = self.analytical_function_Neumann(x, normal_x)
                    self._Neumann_boundary[global_i] += fx
            elif ti.static(self._Q_Neumann == 1):
                for global_vert_idx_i in range(self.num_of_vertices):
                    if self._BEM_manager.get_vertice_type(global_vert_idx_i) == int(VertAttachType.DIRICHLET_TOBESOLVED):
                        x = self._BEM_manager.get_vertice(global_vert_idx_i)
                        normal_x = self._BEM_manager.get_vert_normal(global_vert_idx_i, scope_type=scope_type)
                        fx = self.analytical_function_Neumann(x, normal_x)
                        self._Neumann_boundary[global_vert_idx_i] += fx
        
            if ti.static(self._Q_Dirichlet == 0):
                for local_I in range(self.num_of_panels_Neumann):
                    # Solve Neumann
                    global_i = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_I)
                    # Constant
                    x = ti.Vector([0.0 for j in range(self._dim)], self._ti_dtype)
                    for ii in range(self._dim):
                        x += self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + ii) / float(self._dim)
                    
                    gx = self.analytical_function_Dirichlet(x)
                    self._Dirichlet_boundary[global_i] += gx
            elif ti.static(self._Q_Dirichlet == 1):
                for global_vert_idx_i in range(self.num_of_vertices):
                    if self._BEM_manager.get_vertice_type(global_vert_idx_i) == int(VertAttachType.NEUMANN_TOBESOLVED):
                        x = self._BEM_manager.get_vertice(global_vert_idx_i)
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
    def integrate_on_single_triangle(
        self,
        rands: ti.types.vector(1, int),
        panel_x: int,
        basis_function_index_x: int,
        basis_function_index_y: int,
    ):
        """Get Integration on a single triangle, where
        int_{Tau_x} (func) dx
            = (2 * area_x) * int_{unit_line} (func) (Jacobian) dr1
        
         0

         ============= TO =============

        0---*----*--|1--->x            
          
                                                x2
                                                /
        0---*----*--|1--->x                    /
                                     ->       /
                                             /
                                          x1/
        
         
         - How to project a unit line to a general one?
         - If we choose (0,)->x1, (1,)->x2
         - x (r1, r2) = (1 - r1) * x1 + r2 * x2
        """
        integrand = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)

        area_x = self._BEM_manager.get_panel_area(panel_x)
        
        # Generate number(r1,)
        r1_x = self._BEM_manager.Gauss_points_1d[rands.x]

        # Scale your weight
        weight_x = self._BEM_manager.Gauss_weights_1d[rands.x] * (area_x * 1.0)
        # Get your final weight
        weight = weight_x

        jacobian = 1.0

        phix = self.shape_function(r1_x, -1.0, i=basis_function_index_x)
        phiy = self.shape_function(r1_x, -1.0, i=basis_function_index_y)

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
                panel_type_i = self._BEM_manager.get_panel_type(global_i)
                if panel_type_i == int(CellFluxType.BOTH_TOBESOLVED):
                    panel_type_i = int(CellFluxType.NEUMANN_TOBESOLVED)
                
                for ii in range(basis_func_num_Neumann):
                    for jj in range(basis_func_num_Dirichlet):
                        basis_function_index_x = self._BEM_manager.get_basis_function_index(self._Q_Neumann, ii)
                        basis_function_index_y = self._BEM_manager.get_basis_function_index(self._Q_Dirichlet, jj)

                        local_charge_I = self._BEM_manager.proj_from_local_panel_index_to_local_charge_index(
                            Q_=self._Q_Neumann,
                            local_panel_index=local_I,
                            basis_func_index=basis_function_index_x,
                            panel_type=panel_type_i
                        )

                        global_charge_j = self._BEM_manager.proj_from_local_panel_index_to_global_charge_index(
                            Q_=self._Q_Dirichlet,
                            local_panel_index=local_I,
                            basis_func_index=basis_function_index_y,
                            panel_type=panel_type_i
                        )
                        
                        integrand = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
                        for iii in range(self._GaussQR):
                            rands = ti.Vector([iii], ti.i32)

                            integrand += self.integrate_on_single_triangle(
                                rands=rands,
                                panel_x=global_i,
                                basis_function_index_x=basis_function_index_x, basis_function_index_y=basis_function_index_y
                            )

                        gx = self._Dirichlet_boundary[global_charge_j]

                        if local_charge_I >= 0:
                            if ti.static(self._n == 1):
                                self._rhs_vec[local_charge_I + Neumann_offset_i] += multiplier * integrand * gx
                            elif ti.static(self._n == 2):
                                self._rhs_vec[local_charge_I + Neumann_offset_i] += multiplier * ti.math.cmul(integrand, ti.math.cconj(gx))
            
        if ti.static(self.M_Dirichlet > 0):
            # += 0.5I * f
            for local_I in range(self.num_of_panels_Dirichlet):
                global_i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_I)

                panel_type_i = self._BEM_manager.get_panel_type(global_i)
                if panel_type_i == int(CellFluxType.BOTH_TOBESOLVED):
                    panel_type_i = int(CellFluxType.DIRICHLET_TOBESOLVED)
                
                for ii in range(basis_func_num_Dirichlet):
                    for jj in range(basis_func_num_Neumann):
                        basis_function_index_x = self._BEM_manager.get_basis_function_index(self._Q_Dirichlet, ii)
                        basis_function_index_y = self._BEM_manager.get_basis_function_index(self._Q_Neumann, jj)

                        local_charge_I = self._BEM_manager.proj_from_local_panel_index_to_local_charge_index(
                            Q_=self._Q_Dirichlet,
                            local_panel_index=local_I,
                            basis_func_index=basis_function_index_x,
                            panel_type=panel_type_i
                        )

                        global_charge_j = self._BEM_manager.proj_from_local_panel_index_to_global_charge_index(
                            Q_=self._Q_Neumann,
                            local_panel_index=local_I,
                            basis_func_index=basis_function_index_y,
                            panel_type=panel_type_i
                        )

                        integrand = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
                        for iii in range(self._GaussQR):
                            rands = ti.Vector([iii], ti.i32)

                            integrand += self.integrate_on_single_triangle(
                                rands=rands,
                                panel_x=global_i,
                                basis_function_index_x=basis_function_index_x, basis_function_index_y=basis_function_index_y
                            )

                        fx = self._Neumann_boundary[global_charge_j]

                        if local_charge_I >= 0:
                            if ti.static(self._n == 1):
                                self._rhs_vec[local_charge_I + Dirichlet_offset_i] += multiplier * integrand * fx
                            elif ti.static(self._n == 2):
                                self._rhs_vec[local_charge_I + Dirichlet_offset_i] += multiplier * ti.math.cmul(integrand, ti.math.cconj(fx))
    
    def multiply_M_to_vector(self, scope_type: int, k: float, sqrt_n: float, multiplier: float):
        if self.N_Neumann > 0:
            # += K * g
            self._BEM_manager.double_layer.apply_K_dot_D_boundary(scope_type=scope_type, k=k, sqrt_n=sqrt_n, multiplier=multiplier)
            # -= V * f
            self._BEM_manager.single_layer.apply_V_dot_N_boundary(scope_type=scope_type, k=k, sqrt_n=sqrt_n, multiplier=-multiplier)

        if self.M_Dirichlet > 0:
            # -= K' * f
            self._BEM_manager.adj_double_layer.apply_K_dot_N_boundary(scope_type=scope_type, k=k, sqrt_n=sqrt_n, multiplier=-multiplier)
            # -= W * g
            self._BEM_manager.hypersingular_layer.apply_W_dot_D_boundary(scope_type=scope_type, k=k, sqrt_n=sqrt_n, multiplier=-multiplier)
    
    def forward(self, scope_type: int, k: float, sqrt_n: float):
        self._rhs_vec.fill(0)

        if ti.static(self._BEM_manager._is_transmission <= 0):
            scope_type = self._BEM_manager._scope_type
            if scope_type == int(ScopeType.INTERIOR):
                self.multiply_identity_to_vector(multiplier=0.5)
            elif scope_type == int(ScopeType.EXTERIOR):
                self.multiply_identity_to_vector(multiplier=-0.5)
            else:
                raise RuntimeError("You have to indicate a concrete scope type to construct rhs")

            self.multiply_M_to_vector(scope_type=scope_type, k=k, sqrt_n=sqrt_n, multiplier=1.0)
