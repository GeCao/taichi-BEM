import taichi as ti
import numpy as np

from src.BEM_functions.rhs_constructor import AbstractRHSConstructor
from src.managers.mesh_manager import CellFluxType, VertAttachType


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
        self._k = self._BEM_manager._k
        self._n = self._BEM_manager._n

        self.num_of_Dirichlets = self._BEM_manager.get_num_of_Dirichlets()
        self.num_of_Neumanns = self._BEM_manager.get_num_of_Neumanns()
        self.num_of_vertices = self._BEM_manager.get_num_of_vertices()
        self.num_of_panels = self._BEM_manager.get_num_of_panels()
        self.analyical_function_Dirichlet = self._BEM_manager.analyical_function_Dirichlet
        self.analyical_function_Neumann = self._BEM_manager.analyical_function_Neumann
        self._vert_g_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_vertices,))
        self._panel_f_boundary = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_panels,))
        self._panel_f_vert_g_boundary_compact = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_Neumanns + self.num_of_Dirichlets,))
        self._set_boundaries()
        self._rhs = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=(self.num_of_Dirichlets + self.num_of_Neumanns))
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
        for i in range(self.num_of_panels):
            x1 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * i + 0)
            x2 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * i + 1)
            x3 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * i + 2)

            if self._BEM_manager.get_panel_type(i) == int(CellFluxType.TOBESOLVED):
                # Dirichlet boundary
                vert_idx1 = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * i + 0)
                vert_idx2 = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * i + 1)
                vert_idx3 = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * i + 2)

                self._vert_g_boundary[vert_idx1] = self.analyical_function_Dirichlet(x1)
                self._vert_g_boundary[vert_idx2] = self.analyical_function_Dirichlet(x2)
                self._vert_g_boundary[vert_idx3] = self.analyical_function_Dirichlet(x3)
            elif self._BEM_manager.get_panel_type(i) == int(CellFluxType.NEUMANN_KNOWN):
                # Neumann boundary
                x = (x1 + x2 + x3) / 3.0
                normal_x = self._BEM_manager.get_panel_normal(i)
                self._panel_f_boundary[i] = self.analyical_function_Neumann(x, normal_x)
        
        # num_of_Neumann_panels = self.num_of_panels - self.num_of_Dirichlets
        # Neumann_offset = 0
        # for local_I in range(num_of_Neumann_panels):
        #     global_i = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_I)

        #     self._panel_f_vert_g_boundary_compact[local_I + Neumann_offset] = self._panel_f_boundary[global_i]
        
        # Dirichlet_offset = self.num_of_Neumanns
        # for local_I in range(self.num_of_Dirichlets):
        #     global_i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_I)

        #     global_vert_idx1 = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * global_i + 0)
        #     global_vert_idx2 = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * global_i + 1)
        #     global_vert_idx3 = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * global_i + 2)

        #     local_vert_idx1 = self._BEM_manager.map_global_vert_index_to_local_Neumann(global_vert_idx1)
        #     local_vert_idx2 = self._BEM_manager.map_global_vert_index_to_local_Neumann(global_vert_idx2)
        #     local_vert_idx3 = self._BEM_manager.map_global_vert_index_to_local_Neumann(global_vert_idx3)

        #     self._panel_f_vert_g_boundary_compact[local_vert_idx1 + Dirichlet_offset] = self._vert_g_boundary[global_vert_idx1]
        #     self._panel_f_vert_g_boundary_compact[local_vert_idx2 + Dirichlet_offset] = self._vert_g_boundary[global_vert_idx2]
        #     self._panel_f_vert_g_boundary_compact[local_vert_idx3 + Dirichlet_offset] = self._vert_g_boundary[global_vert_idx3]
    
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
                integrand += 0.5 * (
                    gx
                ) * weight * jacobian
            elif vec_type == int(CellFluxType.NEUMANN_KNOWN):
                phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                fx = self._panel_f_boundary[triangle_x]
                integrand += 0.5 * (
                    fx * phix
                ) * weight * jacobian
            else:
                print("The Cell type should only be Dirichlet or Neumann")
        
        return integrand
    
    @ti.kernel
    def forward(self):
        """
          [           |          ] [   ]
          |-V         | 0.5M + K | | f |
        :=|-----------|----------|*|---|
          | 0.5M - K' |          | | g |
          [           |          ] [   ]
        """
        if ti.static(self.num_of_Dirichlets > 0):
            self._gvec.fill(0)

            # += K * g
            self._BEM_manager.double_layer.apply_K_dot_vert_boundary(
                vert_boundary=self._vert_g_boundary, result_vec=self._gvec, add=True
            )
            # += -V * f
            self._BEM_manager.single_layer.apply_V_dot_panel_boundary(
                panel_boundary=self._panel_f_boundary, result_vec=self._gvec, add=False
            )
        
            # += 0.5M * g
            for I in self._gvec:
                i = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(I)
                self._gvec[I] += self.integrate_on_single_triangle(triangle_x=i, basis_function_index_x=-1)

        if ti.static(self.num_of_Neumanns > 0):
            self._fvec.fill(0)

            # += -K' * f
            self._BEM_manager.adj_double_layer.apply_K_dot_panel_boundary(
                panel_boundary=self._panel_f_boundary, result_vec=self._fvec, add=False
            )
            # += W * g
            self._BEM_manager.hypersingular_layer.apply_W_dot_vert_boundary(
                vert_boundary=self._vert_g_boundary, result_vec=self._fvec, add=True
            )

            # += 0.5M * f
            num_of_Neumann_panels = self.num_of_panels - self.num_of_Dirichlets
            for local_I in range(num_of_Neumann_panels):
                for ii in range(self._dim):
                    global_i = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_I)
                    global_vert_idx = self._BEM_manager.get_vertice_index_from_flat_panel_index(self._dim * global_i + ii)
                    local_vert_idx = self._BEM_manager.map_global_vert_index_to_local_Neumann(global_vert_idx)

                    self._fvec[local_vert_idx] += self.integrate_on_single_triangle(triangle_x=global_i, basis_function_index_x=ii)
