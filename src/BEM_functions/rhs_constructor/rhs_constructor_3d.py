import taichi as ti
import numpy as np

from src.BEM_functions.rhs_constructor import AbstractRHSConstructor
from src.BEM_functions.utils import CellFluxType, VertAttachType, KernelType, PanelsRelation, AssembleType


@ti.data_oriented
class RHSConstructor3d(AbstractRHSConstructor):
    rank = 3

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(RHSConstructor3d, self).__init__(BEM_manager, *args, **kwargs)

        self._GaussQR = self._BEM_manager._GaussQR
        self._Q = self._BEM_manager._Q  # Number of local shape functions

        self._ti_dtype = self._BEM_manager._ti_dtype
        self._np_dtype = self._BEM_manager._np_dtype
        self._dim = self._BEM_manager._dim
        self._kernel_type = self._BEM_manager._kernel_type
        self._n = self._BEM_manager._n

        self.num_of_Dirichlets = self._BEM_manager.get_num_of_Dirichlets()
        self.num_of_Neumanns = self._BEM_manager.get_num_of_Neumanns()
        self.num_of_vertices = self._BEM_manager.get_num_of_vertices()
        self.num_of_panels = self._BEM_manager.get_num_of_panels()

        self.analytical_function_Dirichlet = self._BEM_manager.analytical_function_Dirichlet
        self.analytical_function_Neumann = self._BEM_manager.analytical_function_Neumann

        self._vert_g_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_vertices,))
        self._panel_f_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_panels,))

        self._set_boundaries()

        self._gvec = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=())
        if self.num_of_Dirichlets > 0:
            self._gvec = ti.Vector.field(
                self._n,
                dtype=self._ti_dtype,
                shape=(self.num_of_Dirichlets,)
            )
        self._fvec = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=())
        if self.num_of_Neumanns > 0:
            self._fvec = ti.Vector.field(
                self._n,
                dtype=self._ti_dtype,
                shape=(self.num_of_Neumanns,)
            )

    @ti.kernel
    def _set_boundaries(self):
        sqrt_ni = self._BEM_manager._sqrt_ni
        sqrt_no = self._BEM_manager._sqrt_no

        for i in range(self.num_of_panels):
            x1 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * i + 0)
            x2 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * i + 1)
            x3 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * i + 2)

            if self._BEM_manager.get_panel_type(i) == int(CellFluxType.TOBESOLVED):
                # Dirichlet boundary
                vert_idx1 = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * i + 0)
                vert_idx2 = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * i + 1)
                vert_idx3 = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * i + 2)

                if ti.static(self._kernel_type == int(KernelType.HELMHOLTZ_TRANSMISSION)):
                    if ti.static(self._BEM_manager._use_augment > 0):
                        self._vert_g_boundary[vert_idx1] = (
                            self.analytical_function_Dirichlet(x1, sqrt_ni) - self.analytical_function_Dirichlet(x1, sqrt_no)
                        )
                        self._vert_g_boundary[vert_idx2] = (
                            self.analytical_function_Dirichlet(x2, sqrt_ni) - self.analytical_function_Dirichlet(x2, sqrt_no)
                        )
                        self._vert_g_boundary[vert_idx3] = (
                            self.analytical_function_Dirichlet(x3, sqrt_ni) - self.analytical_function_Dirichlet(x3, sqrt_no)
                        )
                    else:
                        self._vert_g_boundary[vert_idx1] = self.analytical_function_Dirichlet(x1, sqrt_ni)
                        self._vert_g_boundary[vert_idx2] = self.analytical_function_Dirichlet(x1, sqrt_ni)
                        self._vert_g_boundary[vert_idx3] = self.analytical_function_Dirichlet(x1, sqrt_ni)
                else:
                    if ti.static(self._BEM_manager._use_augment > 0):
                        self._vert_g_boundary[vert_idx1] = self.analytical_function_Dirichlet(x1) - self.analytical_function_Dirichlet(x1)
                        self._vert_g_boundary[vert_idx2] = self.analytical_function_Dirichlet(x2) - self.analytical_function_Dirichlet(x2)
                        self._vert_g_boundary[vert_idx3] = self.analytical_function_Dirichlet(x3) - self.analytical_function_Dirichlet(x3)
                    else:
                        self._vert_g_boundary[vert_idx1] = self.analytical_function_Dirichlet(x1)
                        self._vert_g_boundary[vert_idx2] = self.analytical_function_Dirichlet(x2)
                        self._vert_g_boundary[vert_idx3] = self.analytical_function_Dirichlet(x3)
            elif self._BEM_manager.get_panel_type(i) == int(CellFluxType.NEUMANN_KNOWN):
                # Neumann boundary
                x = (x1 + x2 + x3) / 3.0
                normal_x = self._BEM_manager.get_panel_normal(i)
                if ti.static(self._kernel_type == int(KernelType.HELMHOLTZ_TRANSMISSION)):
                    self._panel_f_boundary[i] = (
                        self.analytical_function_Neumann(x, normal_x, sqrt_ni) - self.analytical_function_Neumann(x, normal_x, sqrt_no)
                    )
                else:
                    self._panel_f_boundary[i] = self.analytical_function_Neumann(x, normal_x)
    
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
        self._vert_g_boundary = None
        self._panel_f_boundary = None
    
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

            if vec_type == int(CellFluxType.TOBESOLVED):
                g1 = self._vert_g_boundary[self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * triangle_x + 0)]
                g2 = self._vert_g_boundary[self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * triangle_x + 1)]
                g3 = self._vert_g_boundary[self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * triangle_x + 2)]
                gx = self.interplate_from_unit_triangle_to_general(r1=r1_x, r2=r2_x, x1=g1, x2=g2, x3=g3)
                integrand += (
                    gx
                ) * weight * jacobian
            elif vec_type == int(CellFluxType.NEUMANN_KNOWN):
                phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                fx = self._panel_f_boundary[triangle_x]
                integrand += (
                    fx * phix
                ) * weight * jacobian
            else:
                print("The Cell type should only be Dirichlet or Neumann")
        
        return integrand
    
    @ti.kernel
    def multiply_half_identity_to_vector(self, multiplier: float):
        if ti.static(self.num_of_Dirichlets > 0):
            # += 0.5I * g
            for I in self._gvec:
                i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(I)
                self._gvec[I] += multiplier * self.integrate_on_single_triangle(triangle_x=i, basis_function_index_x=-1)
            
        if ti.static(self.num_of_Neumanns > 0):
            # += 0.5I * f
            num_of_Neumann_panels = self.num_of_panels - self.num_of_Dirichlets
            for local_I in range(num_of_Neumann_panels):
                for ii in range(self._dim):
                    global_i = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_I)
                    global_vert_idx = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * global_i + ii)
                    local_vert_idx = self._BEM_manager.map_global_vert_index_to_local_Neumann(global_vert_idx)

                    self._fvec[local_vert_idx] += multiplier * self.integrate_on_single_triangle(triangle_x=global_i, basis_function_index_x=ii)
    
    def multiply_M_to_vector(self, sqrt_n: float, multiplier: float):
        if self.num_of_Dirichlets > 0:
            # += K * g
            self._BEM_manager.double_layer.apply_K_dot_vert_boundary(sqrt_n=sqrt_n, multiplier=multiplier)
            # -= V * f
            self._BEM_manager.single_layer.apply_V_dot_panel_boundary(sqrt_n=sqrt_n, multiplier=-multiplier)

        if self.num_of_Neumanns > 0:
            # -= K' * f
            self._BEM_manager.adj_double_layer.apply_K_dot_panel_boundary(sqrt_n=sqrt_n, multiplier=-multiplier)
            # -= W * g
            self._BEM_manager.hypersingular_layer.apply_W_dot_vert_boundary(sqrt_n=sqrt_n, multiplier=-multiplier)
    
    def forward(self, assemble_type: int, sqrt_n: float):
        self._fvec.fill(0)
        self._gvec.fill(0)

        self.multiply_half_identity_to_vector(multiplier=0.5)

        if assemble_type == int(AssembleType.ADD_P_MINUS):
            self.multiply_M_to_vector(sqrt_n=sqrt_n, multiplier=-1.0)
        elif assemble_type == int(AssembleType.ADD_P_PLUS):
            self.multiply_M_to_vector(sqrt_n=sqrt_n, multiplier=1.0)
