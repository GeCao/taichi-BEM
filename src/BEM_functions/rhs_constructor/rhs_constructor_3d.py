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

        self.num_of_Dirichlets = self._BEM_manager.get_num_of_Dirichlets()
        self.num_of_Neumanns = self._BEM_manager.get_num_of_Neumanns()
        self.num_of_vertices = self._BEM_manager.get_num_of_vertices()
        self.num_of_panels = self._BEM_manager.get_num_of_panels()

        self.analytical_function_Dirichlet = self._BEM_manager.analytical_function_Dirichlet
        self.analytical_function_Neumann = self._BEM_manager.analytical_function_Neumann

        self._vert_Dirichlet_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_vertices,))
        self._panel_Neumann_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_panels,))

        self._f_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_Dirichlets + self.num_of_Neumanns,))

        self._rhs_vec = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_Dirichlets + self.num_of_Neumanns))

    @ti.kernel
    def set_boundaries(self, sqrt_ni: float, sqrt_no: float):
        Dirichlet_offset_j = self._BEM_manager.get_Dirichlet_offset_j()
        Neumann_offset_j = self._BEM_manager.get_Neumann_offset_j()
        if ti.static(self._BEM_manager._is_transmission > 0):
            for local_I in range(self.num_of_Dirichlets):
                # Dirichlet boundary, solve Neumann
                global_i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_I)
                x1 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 0)
                x2 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 1)
                x3 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 2)

                x = (x1 + x2 + x3) / 3.0
                normal_x = self._BEM_manager.get_panel_normal(global_i)
                fx = self.analytical_function_Neumann(x, normal_x, sqrt_ni) - self.analytical_function_Neumann(x, normal_x, sqrt_no)

                self._f_boundary[Dirichlet_offset_j + local_I] += fx
            
            for vert_index in range(self.num_of_Neumanns):
                local_I = self._BEM_manager.map_global_vert_index_to_local_Neumann(vert_index)
                x = self._BEM_manager.get_vertice(vert_index)
                gx = self.analytical_function_Dirichlet(x, sqrt_ni) - self.analytical_function_Dirichlet(x, sqrt_no)
                if local_I >= 0:
                    self._f_boundary[Neumann_offset_j + local_I] += gx
        else:
            for global_i in range(self.num_of_vertices):
                if self._BEM_manager.get_vertice_type(global_i) == int(VertAttachType.DIRICHLET_KNOWN):
                    # Dirichlet boundary
                    x = self._BEM_manager.get_vertice(global_i)
                    self._vert_Dirichlet_boundary[global_i] = self.analytical_function_Dirichlet(x)

            for global_i in range(self.num_of_panels):
                x1 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 0)
                x2 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 1)
                x3 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * global_i + 2)

                if self._BEM_manager.get_panel_type(global_i) == int(CellFluxType.NEUMANN_KNOWN):
                    # Neumann boundary
                    x = (x1 + x2 + x3) / 3.0
                    normal_x = self._BEM_manager.get_panel_normal(global_i)
                    self._panel_Neumann_boundary[global_i] = self.analytical_function_Neumann(x, normal_x)
    
    @ti.func
    def get_rhs_vec(self):
        return self._rhs_vec
    
    @ti.func
    def get_vert_Dirichlet_boundary(self, vert_index):
        return self._vert_Dirichlet_boundary[vert_index]

    @ti.func
    def get_panel_Neumann_boundary(self, panel_index):
        return self._panel_Neumann_boundary[panel_index]
    
    @ti.func
    def get_f_boundary(self, local_index):
        return self._f_boundary[local_index]

    def kill(self):
        self.num_of_Dirichlets = 0
        self.num_of_Neumanns = 0
        self._rhs_vec = None
        self._vert_Dirichlet_boundary = None
        self._panel_Neumann_boundary = None
    
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
        basis_function_index_x: int
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

        vec_type = self._BEM_manager.get_panel_type(triangle_x)
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

            if vec_type == int(CellFluxType.TOBESOLVED):
                gx = self._vert_Dirichlet_boundary[self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * triangle_x + basis_function_index_x)]
                integrand += (
                    gx * phix
                ) * weight * jacobian
            elif vec_type == int(CellFluxType.NEUMANN_KNOWN):
                fx = self._panel_Neumann_boundary[triangle_x]
                integrand += (
                    fx * phix
                ) * weight * jacobian
            else:
                print("The Cell type should only be Dirichlet or Neumann")
        
        return integrand
    
    @ti.kernel
    def multiply_identity_to_vector(self, multiplier: float):
        if ti.static(self.num_of_Dirichlets > 0):
            # += 0.5I * g
            Dirichlet_offset_i = self._BEM_manager.get_Dirichlet_offset_i()
            for local_I in range(self.num_of_Dirichlets):
                for ii in range(self._dim):
                    global_i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_I)
                    integrand = self.integrate_on_single_triangle(triangle_x=global_i, basis_function_index_x=ii)
                    self._rhs_vec[local_I + Dirichlet_offset_i] += multiplier * integrand
            
        if ti.static(self.num_of_Neumanns > 0):
            # += 0.5I * f
            Neumann_offset_i = self._BEM_manager.get_Neumann_offset_i()
            num_of_Neumann_panels = self.num_of_panels - self.num_of_Dirichlets
            if self.num_of_Neumanns == self.num_of_vertices:
                num_of_Neumann_panels = self.num_of_panels
            for local_I in range(num_of_Neumann_panels):
                for ii in range(self._dim):
                    global_i = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_I)
                    global_vert_idx_i = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * global_i + ii)
                    local_vert_idx_i = self._BEM_manager.map_global_vert_index_to_local_Neumann(global_vert_idx_i)

                    integrand = self.integrate_on_single_triangle(triangle_x=global_i, basis_function_index_x=ii)
                    if local_vert_idx_i >= 0:
                        self._rhs_vec[local_vert_idx_i + Neumann_offset_i] += multiplier * integrand
    
    def multiply_M_to_vector(self, k: float, sqrt_n: float, multiplier: float):
        if self.num_of_Dirichlets > 0:
            # += K * g
            self._BEM_manager.double_layer.apply_K_dot_vert_boundary(k=k, sqrt_n=sqrt_n, multiplier=multiplier)
            # -= V * f
            self._BEM_manager.single_layer.apply_V_dot_panel_boundary(k=k, sqrt_n=sqrt_n, multiplier=-multiplier)

        if self.num_of_Neumanns > 0:
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
