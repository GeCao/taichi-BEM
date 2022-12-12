import os
import numpy as np
import math
import taichi as ti
import torch

from src.BEM_functions.identity_layer import IdentityLayer3d
from src.BEM_functions.single_layer import SingleLayer2d, SingleLayer3d
from src.BEM_functions.double_layer import DoubleLayer2d, DoubleLayer3d
from src.BEM_functions.adj_double_layer import AdjDoubleLayer2d, AdjDoubleLayer3d
from src.BEM_functions.hypersingular_layer import HypersingularLayer2d, HypersingularLayer3d
from src.BEM_functions.rhs_constructor import RHSConstructor3d
from src.BEM_functions.utils import CellFluxType, VertAttachType, KernelType, PanelsRelation, AssembleType
from src.BEM_functions.utils import get_gaussion_integration_points_and_weights, warp_tensor, unwarp_tensor


@ti.data_oriented
class BEMManager:
    def __init__(self, core_manager):
        self._core_manager = core_manager
        self._log_manager = core_manager._log_manager

        self._np_dtype = self._core_manager._np_dtype
        self._ti_dtype = self._core_manager._ti_dtype
        self._kernel_type = int(KernelType.LAPLACE)
        self._is_transmission = self._core_manager._is_transmission
        self._show = -1

        self._simulation_parameters = self._core_manager._simulation_parameters
        if self._simulation_parameters["kernel"] == "Laplace":
            self._kernel_type = int(KernelType.LAPLACE)
            self._n = 1
        elif self._simulation_parameters["kernel"] == "Helmholtz":
            self._kernel_type = int(KernelType.HELMHOLTZ)
            self._n = 2
        elif self._simulation_parameters["kernel"] == "Helmholtz_Transmission":
            self._kernel_type = int(KernelType.HELMHOLTZ_TRANSMISSION)
            self._n = 2
        else:
            raise RuntimeError("Kernel Type only support Laplace and Helmholtz for now")
        
        if self._simulation_parameters["show"] == "Neumann":
            self.show = int(VertAttachType.NEUMANN_TOBESOLVED)
        elif self._simulation_parameters["show"] == "Dirichlet":
            self.show = int(VertAttachType.DIRICHLET_TOBESOLVED)
        elif self._simulation_parameters["show"] == "Default":
            self.show = -1
        else:
            raise RuntimeError("The supported type for visualization is only Neumann/Dirichlet/Default")
        
        if self._show == -1 and self._is_transmission:
            self._show = int(VertAttachType.NEUMANN_TOBESOLVED)
        
        self._k = 0.0
        if self._kernel_type == int(KernelType.HELMHOLTZ) or self._kernel_type == int(KernelType.HELMHOLTZ_TRANSMISSION):
            self._k = self._simulation_parameters["k"]
        self._sqrt_ni = 1.0
        self._sqrt_no = 1.0
        if self._kernel_type == int(KernelType.HELMHOLTZ_TRANSMISSION):
            self._sqrt_ni = float(math.sqrt(self._simulation_parameters["n_i"]))
            self._sqrt_no = float(math.sqrt(self._simulation_parameters["n_o"]))
        
        self.num_of_vertices = 0
        self.num_of_panels = 0

        self._dim = self._simulation_parameters['dim']
        self._Q = self._dim
        self._GaussQR = self._simulation_parameters['GaussQR']

        self.initialized = False

    def initialization(self, analytical_function_Dirichlet, analytical_function_Neumann):
        if not self._core_manager._log_manager.initialized:
            raise RuntimeError("The initialization of Log Manager has the first priority than others!")
        
        if not self._core_manager._mesh_manager.initialized:
            raise RuntimeError("The initialization of Mesh Manager should always before the BEM Manager")
        
        self.num_of_vertices = self._core_manager._mesh_manager.get_num_of_vertices()
        self.num_of_panels = self._core_manager._mesh_manager.get_num_of_panels()

        # Neumann: When Dirichlet/Neumann boundary applied and you want to solve Neumann
        # Default your basis function as constant (Use N_{})
        self.N_Neumann = self._core_manager._mesh_manager.get_N_Neumann()
        self.shape_func_degree_Neumann = self._core_manager._mesh_manager.get_shape_func_degree_Neumann()
        # Dirichlet: When Dirichlet/Neumann boundary applied and you wnat to solve Dirichlet
        # Default your basis function as linear (Use M_{})
        self.M_Dirichlet = self._core_manager._mesh_manager.get_M_Dirichlet()
        self.shape_func_degree_Dirichlet = self._core_manager._mesh_manager.get_shape_func_degree_Dirichlet()

        self.num_of_panels_Neumann = self._core_manager._mesh_manager.get_num_of_panels_Neumann()
        self.num_of_panels_Dirichlet = self._core_manager._mesh_manager.get_num_of_panels_Dirichlet()

        self._Dirichlet_offset_i = 0
        self._Neumann_offset_i = 0
        self._Dirichlet_offset_j = 0
        self._Neumann_offset_j = 0

        self._Dirichlet_offset_i = self.N_Neumann
        self._Neumann_offset_j = self.M_Dirichlet

        self.analytical_function_Dirichlet = analytical_function_Dirichlet
        self.analytical_function_Neumann = analytical_function_Neumann

        # For final comparision
        self.solved = ti.Vector.field(self._n, self._ti_dtype, shape=(self.num_of_vertices, ))
        self.analytical_solved = ti.Vector.field(self._n, self._ti_dtype, shape=(self.num_of_vertices, ))

        # The final global Ax=rhs
        self.raw_solved = ti.Vector.field(self._n, self._ti_dtype, shape=(self.N_Neumann + self.M_Dirichlet, ))
        self.raw_analytical_solved = ti.Vector.field(self._n, self._ti_dtype, shape=(self.N_Neumann + self.M_Dirichlet, ))
        self._mat_A = ti.Vector.field(
            self._n,
            self._ti_dtype,
            shape=(self.N_Neumann + self.M_Dirichlet, self.N_Neumann + self.M_Dirichlet)
        )
        self.rhs = ti.Vector.field(self._n, self._ti_dtype, shape=(self.N_Neumann + self.M_Dirichlet, ))
        self._mat_P = ti.Vector.field(self._n, self._ti_dtype, shape=())
        if self._is_transmission > 0:
            self.rhs = ti.Vector.field(self._n, self._ti_dtype, shape=(2 * (self.N_Neumann + self.M_Dirichlet), ))
            self._mat_P = ti.Vector.field(
                self._n,
                self._ti_dtype,
                shape=(self.N_Neumann + self.M_Dirichlet, self.N_Neumann + self.M_Dirichlet)
            )

        # For visualization
        self.solved_vert_color = ti.Vector.field(self._dim, ti.f32, shape=(self.num_of_vertices, ))
        self.solved_vertices = ti.Vector.field(self._dim, ti.f32, shape=(self.num_of_vertices, ))
        self.analytical_vert_color = ti.Vector.field(self._dim, ti.f32, shape=(self.num_of_vertices, ))
        self.analytical_vertices = ti.Vector.field(self._dim, ti.f32, shape=(self.num_of_vertices, ))
        self.diff_vert_color = ti.Vector.field(self._dim, ti.f32, shape=(self.num_of_vertices, ))
        self.diff_vertices = ti.Vector.field(self._dim, ti.f32, shape=(self.num_of_vertices, ))

        # For numerical integrations
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

        # Prepare BIOs
        if self._dim == 2:
            self.single_layer = SingleLayer2d(self)
            self.double_layer = DoubleLayer2d(self)
            self.adj_double_layer = AdjDoubleLayer2d(self)
            self.hypersingular_layer = HypersingularLayer2d(self)
        elif self._dim == 3:
            self.identity_layer = IdentityLayer3d(self)
            self.single_layer = SingleLayer3d(self)
            self.double_layer = DoubleLayer3d(self)
            self.adj_double_layer = AdjDoubleLayer3d(self)
            self.hypersingular_layer = HypersingularLayer3d(self)

            self.rhs_constructor = RHSConstructor3d(self)
        else:
            raise RuntimeError("The dimension of our scene should only be 2D or 3D")

        self.initialized = True
    
    def get_GaussQR(self):
        return self._GaussQR
    
    def get_num_of_vertices(self):
        return self.num_of_vertices
    
    def get_num_of_panels(self):
        return self.num_of_panels

    def get_M_Dirichlet(self):
        return self.M_Dirichlet
    
    def get_N_Neumann(self):
        return self.N_Neumann

    def get_num_of_panels_Neumann(self):
        return self.num_of_panels_Neumann
    
    def get_num_of_panels_Dirichlet(self):
        return self.num_of_panels_Dirichlet
    
    def get_shape_func_degree_Neumann(self):
        return self.shape_func_degree_Neumann
    
    def get_shape_func_degree_Dirichlet(self):
        return self.shape_func_degree_Dirichlet
    
    @ti.func
    def get_num_of_basis_functions_from_Q(self, Q_):
        result = -1
        if Q_ == 0:
            # constant
            result = 1
        elif Q_ == 1:
            # Linear
            result = self._dim
        
        return result
    
    @ti.func
    def get_basis_function_index(self, Q_, ii):
        result = -1
        if Q_ == 1:
            result = ii
        
        return result
    
    @ti.func
    def proj_from_local_panel_index_to_local_charge_index(
        self,
        Q_,
        local_panel_index,
        basis_func_index,
        panel_type: int,
    ):
        result = -1

        if Q_ == 0:
            # Constant on faces
            result = local_panel_index
        elif Q_ == 1:
            # Linear on vertices
            global_panel_I = -1
            global_vert_i = -1
            if panel_type == int(CellFluxType.NEUMANN_TOBESOLVED):
                global_panel_I = self.map_local_Neumann_index_to_panel_index(local_panel_index)
                global_vert_i = self.get_vertice_index_from_flat_panel_index(self._dim * global_panel_I + basis_func_index)
                local_vert_i = self.map_global_vert_index_to_local_Neumann(global_vert_i)
                result = local_vert_i
            elif panel_type == int(CellFluxType.DIRICHLET_TOBESOLVED):
                global_panel_I = self.map_local_Dirichlet_index_to_panel_index(local_panel_index)
                global_vert_i = self.get_vertice_index_from_flat_panel_index(self._dim * global_panel_I + basis_func_index)
                local_vert_i = self.map_global_vert_index_to_local_Dirichlet(global_vert_i)
                result = local_vert_i
        
        return result
    
    @ti.func
    def proj_from_local_panel_index_to_global_charge_index(
        self,
        Q_,
        local_panel_index,
        basis_func_index,
        panel_type: int,
    ):
        result = -1

        if Q_ == 0:
            # Constant on faces
            if panel_type == int(CellFluxType.NEUMANN_TOBESOLVED):
                global_panel_I = self.map_local_Neumann_index_to_panel_index(local_panel_index)
                result = global_panel_I
            elif panel_type == int(CellFluxType.DIRICHLET_TOBESOLVED):
                global_panel_I = self.map_local_Dirichlet_index_to_panel_index(local_panel_index)
                result = global_panel_I
        elif Q_ == 1:
            # Linear on vertices
            global_panel_I = -1
            global_vert_i = -1
            if panel_type == int(CellFluxType.NEUMANN_TOBESOLVED):
                global_panel_I = self.map_local_Neumann_index_to_panel_index(local_panel_index)
                global_vert_i = self.get_vertice_index_from_flat_panel_index(self._dim * global_panel_I + basis_func_index)
                result = global_vert_i
            elif panel_type == int(CellFluxType.DIRICHLET_TOBESOLVED):
                global_panel_I = self.map_local_Dirichlet_index_to_panel_index(local_panel_index)
                global_vert_i = self.get_vertice_index_from_flat_panel_index(self._dim * global_panel_I + basis_func_index)
                result = global_vert_i
        
        return result

    @ti.func
    def get_vertice(self, vert_index):
        return self._core_manager._mesh_manager.vertices[vert_index]
    
    @ti.func
    def get_vertice_index_from_flat_panel_index(self, panel_index):
        return self._core_manager._mesh_manager.panels[panel_index]
    
    @ti.func
    def get_vertice_from_flat_panel_index(self, panel_index):
        vert_index = self.get_vertice_index_from_flat_panel_index(panel_index)
        return self.get_vertice(vert_index)
    
    @ti.func
    def get_panel_area(self, panel_area_index):
        return self._core_manager._mesh_manager.get_panel_areas()[panel_area_index]
    
    @ti.func
    def get_panel_normal(self, panel_normal_index):
        return self._core_manager._mesh_manager.get_panel_normals()[panel_normal_index]

    @ti.func
    def get_vert_normal(self, vert_index):
        return self._core_manager._mesh_manager.get_vert_normals()[vert_index]
    
    @ti.func
    def get_vert_area(self, vert_index):
        return self._core_manager._mesh_manager.get_vert_areas()[vert_index]
    
    @ti.func
    def map_local_Dirichlet_index_to_panel_index(self, local_index):
        return self._core_manager._mesh_manager.map_local_Dirichlet_index_to_panel_index(local_index)
    
    @ti.func
    def map_local_Neumann_index_to_panel_index(self, local_index):
        return self._core_manager._mesh_manager.map_local_Neumann_index_to_panel_index(local_index)
    
    @ti.func
    def map_global_vert_index_to_local_Dirichlet(self, vert_index):
        return self._core_manager._mesh_manager.map_global_vert_index_to_local_Dirichlet(vert_index)
    
    @ti.func
    def map_global_vert_index_to_local_Neumann(self, vert_index):
        return self._core_manager._mesh_manager.map_global_vert_index_to_local_Neumann(vert_index)
    
    @ti.func
    def get_panels_relation(self, i, j):
        return self._core_manager._mesh_manager.get_panels_relation(i, j)
    
    @ti.func
    def get_panel_type(self, panel_index):
        return self._core_manager._mesh_manager.get_panel_types()[panel_index]
    
    @ti.func
    def get_vertice_type(self, vert_index):
        return self._core_manager._mesh_manager.get_vertice_types()[vert_index]
    
    @ti.func
    def get_mat_A(self):
        return self._mat_A
    
    @ti.func
    def get_mat_P(self):
        return self._mat_P
    
    @ti.func
    def get_Dirichlet_offset_i(self):
        return self._Dirichlet_offset_i
    
    @ti.func
    def get_Dirichlet_offset_j(self):
        return self._Dirichlet_offset_j
    
    @ti.func
    def get_Neumann_offset_i(self):
        return self._Neumann_offset_i
    
    @ti.func
    def get_Neumann_offset_j(self):
        return self._Neumann_offset_j
    
    @ti.func
    def shape_function(self, r1, r2, i: int):
        """
        Providing x (r1, r2) = (1 - r1) * x1 + (r1 - r2) * x2 + r2 * x3
        """
        # Default as 1 to cancel shape function term
        result = 0.0 * r1 + 1.0  # Avoid type check, this equals as: result = 1.0
        if i == 0:
            result = 1 - r1
        elif i == 1:
            result = r1 - r2
        elif i == 2:
            result = r2
        return result
    
    @ti.func
    def G(self, x, y, k, sqrt_n: float = 1):
        Gxy = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
        distance = ti.Vector.norm(x - y)
        if ti.static(self._kernel_type == int(KernelType.LAPLACE)):
            Gxy.x = (1.0 / 4.0 / ti.math.pi) / distance
        elif ti.static(self._kernel_type == int(KernelType.HELMHOLTZ)):
            Gxy = (1.0 / 4.0 / ti.math.pi) / distance * ti.math.cexp(
                ti.Vector([0.0, k * distance], self._ti_dtype)
            )
        elif ti.static(self._kernel_type == int(KernelType.HELMHOLTZ_TRANSMISSION)):
            Gxy = (1.0 / 4.0 / ti.math.pi) / distance * ti.math.cexp(
                ti.Vector([0.0, k * sqrt_n * distance], self._ti_dtype)
            )
        
        return Gxy
    
    @ti.func
    def grad_G_y(self, x, y, normal_y, k, sqrt_n: float = 1):
        grad_Gy = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
        direc = ti.Vector.normalized(x - y)
        distance = ti.Vector.norm(x - y)
        if ti.static(self._kernel_type == int(KernelType.LAPLACE)):
            grad_Gy.x = (1.0 / 4.0 / ti.math.pi) * ti.math.dot(direc, normal_y) / ti.math.pow(distance, 2)
        elif ti.static(self._kernel_type == int(KernelType.HELMHOLTZ)):
            chained_complex_vector = ti.math._complex.cmul(
                ti.Vector([1.0, -k * distance], self._ti_dtype),
                ti.math.cexp(ti.Vector([0.0, k * distance], self._ti_dtype))
            )
            grad_Gy = (1.0 / 4.0 / ti.math.pi) * ti.math.dot(direc, normal_y) / ti.math.pow(distance, 2) * chained_complex_vector
        elif ti.static(self._kernel_type == int(KernelType.HELMHOLTZ_TRANSMISSION)):
            chained_complex_vector = ti.math._complex.cmul(
                ti.Vector([1.0, -k * sqrt_n * distance], self._ti_dtype),
                ti.math.cexp(ti.Vector([0.0, k * sqrt_n * distance], self._ti_dtype))
            )
            grad_Gy = (1.0 / 4.0 / ti.math.pi) * ti.math.dot(direc, normal_y) / ti.math.pow(distance, 2) * chained_complex_vector
        
        return grad_Gy
    
    @ti.kernel
    def compute_color(self):
        # Please Note, if you are applying Neumann boundary, and want to get Dirichlet solve
        # since f'(x) = [f(x) + C]', where C is a constant number,
        # our solved Dirichlet usually does not hold the same scales as the analytical solves
        # In this case, scale our Dirichlet solve to the analytical Dirichlets

        self.solved_vert_color.fill(0)
        self.analytical_vert_color.fill(0)
        self.diff_vert_color.fill(0)

        max_analytical_Neumann_solved = -5e11
        max_analytical_Dirichlet_solved = -5e11
        mean_solved_Dirichlet = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
        mean_analytical_Dirichlet_solved = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)

        num_of_vertices_Dirichlet = 0
        for i in range(self.num_of_vertices):
            if self.get_vertice_type(i) != int(VertAttachType.DIRICHLET_TOBESOLVED):
                ti.atomic_max(max_analytical_Neumann_solved, ti.Vector.norm(self.analytical_solved[i]))
            elif self.get_vertice_type(i) != int(VertAttachType.NEUMANN_TOBESOLVED):
                num_of_vertices_Dirichlet += 1
                mean_solved_Dirichlet += self.solved[i]
                mean_analytical_Dirichlet_solved += self.analytical_solved[i]
                ti.atomic_max(max_analytical_Dirichlet_solved, ti.Vector.norm(self.analytical_solved[i]))
        if num_of_vertices_Dirichlet > 0:
            mean_solved_Dirichlet /= num_of_vertices_Dirichlet
            mean_analytical_Dirichlet_solved /= num_of_vertices_Dirichlet
        
        # Do your scale:
        for i in range(self.num_of_vertices):
            if self.get_vertice_type(i) == int(VertAttachType.DIRICHLET_TOBESOLVED) or \
                self._show == int(VertAttachType.DIRICHLET_TOBESOLVED):
                self.solved[i] += mean_analytical_Dirichlet_solved - mean_solved_Dirichlet
                
        for i in range(self.num_of_vertices):
            if self.get_vertice_type(i) != int(VertAttachType.DIRICHLET_TOBESOLVED) and self._show != int(VertAttachType.DIRICHLET_TOBESOLVED):
                if self.analytical_solved[i].x > 0:
                    self.analytical_vert_color[i].x = ti.Vector.norm(self.analytical_solved[i]) / max_analytical_Neumann_solved
                else:
                    self.analytical_vert_color[i].z = ti.Vector.norm(self.analytical_solved[i]) / max_analytical_Neumann_solved
                
                if self.solved[i].x > 0:
                    self.solved_vert_color[i].x = ti.Vector.norm(self.solved[i]) / max_analytical_Neumann_solved
                else:
                    self.solved_vert_color[i].z = ti.Vector.norm(self.solved[i]) / max_analytical_Neumann_solved
                
            elif self.get_vertice_type(i) != int(VertAttachType.NEUMANN_TOBESOLVED) or self._show != int(VertAttachType.NEUMANN_TOBESOLVED):
                if self.analytical_solved[i].x > 0:
                    self.analytical_vert_color[i].x = ti.Vector.norm(self.analytical_solved[i]) / max_analytical_Dirichlet_solved
                else:
                    self.analytical_vert_color[i].z = ti.Vector.norm(self.analytical_solved[i]) / max_analytical_Dirichlet_solved
                
                if self.solved[i].x > 0:
                    self.solved_vert_color[i].x = ti.Vector.norm(self.solved[i]) / max_analytical_Dirichlet_solved
                else:
                    self.solved_vert_color[i].z = ti.Vector.norm(self.solved[i]) / max_analytical_Dirichlet_solved
            
            self.diff_vert_color[i].y = ti.abs(
                self.solved_vert_color[i].x - self.analytical_vert_color[i].x - self.solved_vert_color[i].z + self.analytical_vert_color[i].z
            )
        
    @ti.kernel
    def set_mesh_instances(self):  
        for I in self.solved_vertices:
            self.analytical_vertices[I] = self.get_vertice(I)
            self.analytical_vertices[I].z += 2.2

            self.solved_vertices[I] = self.get_vertice(I)

            self.diff_vertices[I] = self.get_vertice(I)
            self.diff_vertices[I].z -= 2.2

    @ti.kernel
    def matA_add_M(self, multiplier: float):
        # += M
        if ti.static(self.N_Neumann > 0):
            for I, J in self.single_layer._Vmat:
                self._mat_A[I + self._Neumann_offset_i, J + self._Neumann_offset_j] += -multiplier * self.single_layer._Vmat[I, J]
        
        if ti.static(self.N_Neumann * self.M_Dirichlet > 0):
            for I, J in self.double_layer._Kmat:
                self._mat_A[I + self._Neumann_offset_i, J + self._Dirichlet_offset_j] += multiplier * self.double_layer._Kmat[I, J]
        
        if ti.static(self.M_Dirichlet * self.N_Neumann > 0):
            for I in range(self.N_Neumann):
                for J in range(self.M_Dirichlet):
                    self._mat_A[J + self._Dirichlet_offset_i, I + self._Neumann_offset_j] += -multiplier * self.double_layer._Kmat[I, J]
            # for I, J in self.adj_double_layer._Kmat:
            #     self._mat_A[I + self._Dirichlet_offset_i, J + self._Neumann_offset_j] += -multiplier * self.adj_double_layer._Kmat[I, J]
        
        if ti.static(self.M_Dirichlet > 0):
            for I, J in self.hypersingular_layer._Wmat:
                self._mat_A[I + self._Dirichlet_offset_i, J + self._Dirichlet_offset_j] += -multiplier * self.hypersingular_layer._Wmat[I, J]

    def assemble_matA(self, assemble_type: int, multiplier: float):
        """
        M = [K,  -V ]
            [-W, -K']
        """
        if assemble_type == int(AssembleType.ADD_M):
            # += M
            self.matA_add_M(multiplier)
        elif assemble_type == int(AssembleType.ADD_HALF_IDENTITY):
            self.identity_layer.matA_add_global_Identity_matrix(0.5 * multiplier)
        elif assemble_type == int(AssembleType.ADD_P_MINUS):
            self.identity_layer.matA_add_global_Identity_matrix(0.5 * multiplier)
            self.matA_add_M(-multiplier)
        elif assemble_type == int(AssembleType.ADD_P_PLUS):
            self.identity_layer.matA_add_global_Identity_matrix(0.5 * multiplier)
            self.matA_add_M(multiplier)
    
    @ti.kernel
    def matP_add_M(self, multiplier: float):
        # += M
        if ti.static(self.N_Neumann > 0):
            for I, J in self.single_layer._Vmat:
                self._mat_P[I + self._Neumann_offset_i, J + self._Neumann_offset_j] += -multiplier * self.single_layer._Vmat[I, J]
        
        if ti.static(self.N_Neumann * self.M_Dirichlet > 0):
            for I, J in self.double_layer._Kmat:
                self._mat_P[I + self._Neumann_offset_i, J + self._Dirichlet_offset_j] += multiplier * self.double_layer._Kmat[I, J]
        
        if ti.static(self.M_Dirichlet * self.N_Neumann > 0):
            for I in range(self.N_Neumann):
                for J in range(self.M_Dirichlet):
                    self._mat_P[J + self._Dirichlet_offset_i, I + self._Neumann_offset_j] += -multiplier * self.double_layer._Kmat[I, J]
            # for I, J in self.adj_double_layer._Kmat:
            #     self._mat_P[I + self._Dirichlet_offset_i, J + self._Neumann_offset_j] += -multiplier * self.adj_double_layer._Kmat[I, J]
        
        if ti.static(self.M_Dirichlet > 0):
            for I, J in self.hypersingular_layer._Wmat:
                self._mat_P[I + self._Dirichlet_offset_i, J + self._Dirichlet_offset_j] += -multiplier * self.hypersingular_layer._Wmat[I, J]
    
    def assemble_matP(self, assemble_type: int, multiplier: float):
        """
        M = [K,  -V ]
            [-W, -K']
        """
        if assemble_type == int(AssembleType.ADD_M):
            # += M
            self.matP_add_M(multiplier)
        elif assemble_type == int(AssembleType.ADD_HALF_IDENTITY):
            self.identity_layer.matP_add_global_Identity_matrix(0.5 * multiplier)
        elif assemble_type == int(AssembleType.ADD_P_MINUS):
            self.identity_layer.matP_add_global_Identity_matrix(0.5 * multiplier)
            self.matP_add_M(-multiplier)
        elif assemble_type == int(AssembleType.ADD_P_PLUS):
            self.identity_layer.matP_add_global_Identity_matrix(0.5 * multiplier)
            self.matP_add_M(multiplier)
            
    @ti.kernel
    def assemble_rhs(self):
        if ti.static(self._is_transmission):
            self.rhs.fill(0)
            for I, J in self._mat_P:
                if ti.static(self._n == 1):
                    self.rhs[I] += self._mat_P[I, J] * self.rhs_constructor.get_f_boundary(J)
                elif ti.static(self._n == 2):
                    self.rhs[I] += ti.math.cmul(
                        self._mat_P[I, J], ti.math.cconj(self.rhs_constructor.get_f_boundary(J))
                    )
        else:
            for I in self.rhs:
                self.rhs[I] = self.rhs_constructor.get_rhs_vec()[I]
        
    @ti.kernel
    def compute_analytical_raw_solved(self):
        self.raw_analytical_solved.fill(0)

        if ti.static(self._is_transmission > 0):
            # rhs = P_O^- * f       
            # for I in self.raw_analytical_solved:
            #     self.raw_analytical_solved[I] = self.rhs[I]
            
            for I in self.rhs_constructor._Dirichlet_boundary:
                self.raw_analytical_solved[I + self._Dirichlet_offset_j] = self.rhs_constructor._Dirichlet_boundary[I]
            
            for I in self.rhs_constructor._Neumann_boundary:
                self.raw_analytical_solved[I + self._Neumann_offset_j] = self.rhs_constructor._Neumann_boundary[I]
        else:
            Dirichlet_offset = self._Dirichlet_offset_j
            Neumann_offset = self._Neumann_offset_j
            if ti.static(self.N_Neumann > 0):
                if ti.static(self.shape_func_degree_Neumann == 0):
                    for local_I in range(self.num_of_panels_Neumann):
                        # Solve Neumann
                        global_i = self.map_local_Neumann_index_to_panel_index(local_I)
                        x1 = self.get_vertice_from_flat_panel_index(self._dim * global_i + 0)
                        x2 = self.get_vertice_from_flat_panel_index(self._dim * global_i + 1)
                        x3 = self.get_vertice_from_flat_panel_index(self._dim * global_i + 2)
                        x = (x1 + x2 + x3) / 3.0
                        normal_x = self.get_panel_normal(global_i)
                        fx = self.rhs_constructor.analytical_function_Neumann(x, normal_x)
                        self.raw_analytical_solved[local_I + Neumann_offset] = fx
                        self.raw_solved[local_I + Neumann_offset] += self.rhs_constructor.get_Neumann_boundary(local_I, -1)
                elif ti.static(self.shape_func_degree_Neumann == 1):
                    for global_vert_i in range(self.num_of_vertices):
                        # Solve Neumann
                        if self.get_vertice_type(global_vert_i) == int(VertAttachType.NEUMANN_TOBESOLVED):
                            local_vert_i = self.map_global_vert_index_to_local_Neumann(global_vert_i)
                            x = self.get_vertice(global_vert_i)
                            normal_x = self.get_vert_normal(global_vert_i)
                            fx = self.rhs_constructor.analytical_function_Neumann(x, normal_x)
                            if local_vert_i >= 0:
                                self.raw_analytical_solved[local_vert_i + Neumann_offset] = fx
                                self.raw_solved[local_vert_i + Neumann_offset] += self.rhs_constructor._Neumann_boundary[global_vert_i]
            
            if ti.static(self.M_Dirichlet > 0):
                if ti.static(self.shape_func_degree_Dirichlet == 0):
                    for local_I in range(self.num_of_panels_Dirichlet):
                        # Solve Dirichlet
                        global_i = self.map_local_Dirichlet_index_to_panel_index(local_I)
                        x1 = self.get_vertice_from_flat_panel_index(self._dim * global_i + 0)
                        x2 = self.get_vertice_from_flat_panel_index(self._dim * global_i + 1)
                        x3 = self.get_vertice_from_flat_panel_index(self._dim * global_i + 2)
                        x = (x1 + x2 + x3) / 3.0
                        normal_x = self.get_panel_normal(global_i)
                        gx = self.rhs_constructor.analytical_function_Dirichlet(x)
                        self.raw_analytical_solved[local_I + Dirichlet_offset] = gx
                        self.raw_solved[local_I + Dirichlet_offset] += self.rhs_constructor.get_Dirichlet_boundary(local_I, -1)
                elif ti.static(self.shape_func_degree_Dirichlet == 1):
                    for global_vert_i in range(self.num_of_vertices):
                        # Solve Dirichlet
                        if self.get_vertice_type(global_vert_i) == int(VertAttachType.DIRICHLET_TOBESOLVED):
                            local_vert_i = self.map_global_vert_index_to_local_Dirichlet(global_vert_i)
                            x = self.get_vertice(global_vert_i)
                            normal_x = self.get_vert_normal(global_vert_i)
                            gx = self.rhs_constructor.analytical_function_Dirichlet(x)
                            if local_vert_i >= 0:
                                self.raw_analytical_solved[local_vert_i + Dirichlet_offset] = gx
                                self.raw_solved[local_vert_i + Dirichlet_offset] += self.rhs_constructor._Dirichlet_boundary[global_vert_i]
    
    @ti.kernel
    def splat_u_from_raw_solved_to_vertices(self):
        self.solved.fill(0)
        self.analytical_solved.fill(0)
        
        if ti.static(self._dim == 3):
            Dirichlet_offset = self._Dirichlet_offset_j
            Neumann_offset = self._Neumann_offset_j
            if self._is_transmission:
                Dirichlet_offset = self._Dirichlet_offset_j
                Neumann_offset = self._Neumann_offset_j
            
            if ti.static(self._show != int(VertAttachType.DIRICHLET_TOBESOLVED)):
                if ti.static(self.shape_func_degree_Neumann == 0):
                    for I in range(self.num_of_panels_Neumann):
                        global_i = self.map_local_Neumann_index_to_panel_index(I)
                        area_x = self.get_panel_area(global_i)
                        for ii in range(self._dim):
                            global_vert_idx_i = self.get_vertice_index_from_flat_panel_index(self._dim * global_i + ii)
                            area_vert_i = self.get_vert_area(global_vert_idx_i)

                            if self.get_panel_type(global_i) != int(CellFluxType.DIRICHLET_TOBESOLVED):
                                self.solved[global_vert_idx_i] += self.raw_solved[Neumann_offset + I] * area_x / 3.0 / area_vert_i
                                self.analytical_solved[global_vert_idx_i] += self.raw_analytical_solved[Neumann_offset + I] * area_x / 3.0 / area_vert_i
                elif ti.static(self.shape_func_degree_Neumann == 1):
                    for global_vert_i in range(self.num_of_vertices):
                        if self.get_vertice_type(global_vert_i) != int(VertAttachType.DIRICHLET_TOBESOLVED):
                            local_I = self.map_global_vert_index_to_local_Neumann(global_vert_i)
                            if local_I >= 0:
                                self.solved[global_vert_i] += self.raw_solved[Neumann_offset + local_I]
                                self.analytical_solved[global_vert_i] += self.raw_analytical_solved[Neumann_offset + local_I]
            
            if ti.static(self._show != int(VertAttachType.NEUMANN_TOBESOLVED)):
                if ti.static(self.shape_func_degree_Dirichlet == 0):
                    for I in range(self.num_of_panels_Dirichlet):
                        global_i = self.map_local_Dirichlet_index_to_panel_index(I)
                        area_x = self.get_panel_area(global_i)
                        for ii in range(self._dim):
                            global_vert_idx_i = self.get_vertice_index_from_flat_panel_index(self._dim * global_i + ii)
                            area_vert_i = self.get_vert_area(global_vert_idx_i)

                            if self.get_panel_type(global_i) != int(CellFluxType.NEUMANN_TOBESOLVED):
                                self.solved[global_vert_idx_i] += self.raw_solved[Dirichlet_offset + I] * area_x / 3.0 / area_vert_i
                                self.analytical_solved[global_vert_idx_i] += self.raw_analytical_solved[Dirichlet_offset + I] * area_x / 3.0 / area_vert_i
                elif ti.static(self.shape_func_degree_Dirichlet == 1):
                    for global_vert_i in range(self.num_of_vertices):
                        if self.get_vertice_type(global_vert_i) != int(VertAttachType.NEUMANN_TOBESOLVED):
                            local_I = self.map_global_vert_index_to_local_Dirichlet(global_vert_i)
                            if local_I >= 0:
                                self.solved[global_vert_i] += self.raw_solved[Dirichlet_offset + local_I]
                                self.analytical_solved[global_vert_i] += self.raw_analytical_solved[Dirichlet_offset + local_I]

    @ti.kernel
    def Jacobian_solver(self) -> float:
        residual = 0.0
        for I in self.raw_solved:
            new_u = self.rhs[I] + self._mat_A[I, I] * self.raw_solved[I]
            for J in range(self.raw_solved.shape[0]):
                new_u -= self._mat_A[I, J] * self.raw_solved[J]
            new_u /= self._mat_A[I, I]
            # residual += (self.raw_solved[I] - new_u) * (self.raw_solved[I] - new_u)
            self.raw_solved[I] = new_u
        
        for I in self.raw_solved:
            b_Ax = self.rhs[I]
            for J in range(self.raw_solved.shape[0]):
                b_Ax -= self._mat_A[I, J] * self.raw_solved[J]
            residual += b_Ax * b_Ax
        
        residual = ti.math.sqrt(residual / self.raw_solved.shape[0])
        return residual
    
    def matrix_layer_init(self):
        self.single_layer._Vmat.fill(0)
        self.double_layer._Kmat.fill(0)
        self.adj_double_layer._Kmat.fill(0)
        self.hypersingular_layer._Wmat.fill(0)

    def matrix_layer_forward(self, k: float, sqrt_n: float = 1.0):
        self.single_layer.forward(k, sqrt_n)
        self.double_layer.forward(k, sqrt_n)
        self.adj_double_layer.forward(k, sqrt_n)
        self.hypersingular_layer.forward(k, sqrt_n)
        self._log_manager.InfoLog("Construct all Matrix Layers Done")
    
    def rhs_layer_forward(self, assemble_type: int, k: float, sqrt_n: float = 1.0):
        self.rhs_constructor.forward(assemble_type=assemble_type, k=k, sqrt_n=sqrt_n)
        self._log_manager.InfoLog("Construct all RHS Layers Done")
    
    def compute_norm(self, A: torch.Tensor):
        # Do your svd firstly
        S = torch.linalg.svdvals(A)
        A_norm = S.max()

        return A_norm.item()
    
    def compute_inv_norm(self, A: torch.Tensor):
        # Do your svd firstly
        scale_A = 100.0
        A = scale_A * A
        
        S = torch.linalg.svdvals(A)

        inv_S = 1.0 / S
        inv_S = (inv_S) * scale_A
        A_norm = inv_S.max()

        return A_norm.item()
    
    def get_mat_A1_inv_norm(self, k: float):
        self._mat_A.fill(0)  # A1

        self.matrix_layer_forward(k=k, sqrt_n=self._sqrt_ni)
        self.assemble_matA(assemble_type=int(AssembleType.ADD_P_PLUS), multiplier=-1.0)  # Reduce P_PLUS_I
        self.matrix_layer_forward(k=k, sqrt_n=self._sqrt_no)
        self.assemble_matA(assemble_type=int(AssembleType.ADD_P_MINUS), multiplier=1.0)  # Add P_MINUS_O

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch_mat_A1 = warp_tensor(self._mat_A.to_torch().to(device))
        mat_A_inv_norm1 = self.compute_inv_norm(torch_mat_A1)

        return mat_A_inv_norm1
    
    def get_mat_A2_inv_norm(self, k: float):
        self._mat_A.fill(0)

        self.matrix_layer_forward(k=k, sqrt_n=self._sqrt_ni)
        self.assemble_matA(assemble_type=int(AssembleType.ADD_P_PLUS), multiplier=1.0)  # Add P_PLUS_I
        self.matrix_layer_forward(k=k, sqrt_n=self._sqrt_no)
        self.assemble_matA(assemble_type=int(AssembleType.ADD_P_MINUS), multiplier=1.0)  # Add P_MINUS_O

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_mat_A2 = warp_tensor(self._mat_A.to_torch().to(device))
        mat_A_inv_norm2 = self.compute_inv_norm(torch_mat_A2)

        return mat_A_inv_norm2
    
    def get_augment_mat_A1_inv_norm(self, k: float):
        self._mat_A.fill(0)
        self._mat_P.fill(0)

        self.matrix_layer_forward(k=k, sqrt_n=self._sqrt_ni)
        self.assemble_matA(assemble_type=int(AssembleType.ADD_P_PLUS), multiplier=-1.0)  # Reduce P_PLUS_I
        self.assemble_matP(assemble_type=int(AssembleType.ADD_P_PLUS), multiplier=1.0)  # ADD P_PLUS_I
        self.matrix_layer_forward(k=k, sqrt_n=self._sqrt_no)
        self.assemble_matA(assemble_type=int(AssembleType.ADD_P_MINUS), multiplier=1.0)  # Add P_MINUS_O

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_augmented_A = torch.cat(
            (warp_tensor(self._mat_A.to_torch().to(device)),
             warp_tensor(self._mat_P.to_torch().to(device))),
            0
        )
        mat_A_norm_inv = self.compute_inv_norm(torch_augmented_A)

        return mat_A_norm_inv
    
    def get_augment_mat_A2_inv_norm(self, k: float):
        self._mat_A.fill(0)
        self._mat_P.fill(0)

        self.matrix_layer_forward(k=k, sqrt_n=self._sqrt_ni)
        self.assemble_matA(assemble_type=int(AssembleType.ADD_P_PLUS), multiplier=1.0)  # Add P_PLUS_I
        self.assemble_matP(assemble_type=int(AssembleType.ADD_P_PLUS), multiplier=1.0)  # ADD P_PLUS_I
        self.matrix_layer_forward(k=k, sqrt_n=self._sqrt_no)
        self.assemble_matA(assemble_type=int(AssembleType.ADD_P_MINUS), multiplier=1.0)  # Add P_MINUS_O

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_augmented_A = torch.cat(
            (warp_tensor(self._mat_A.to_torch().to(device)),
             warp_tensor(self._mat_P.to_torch().to(device))),
            0
        )
        mat_A_norm_inv = self.compute_inv_norm(torch_augmented_A)

        return mat_A_norm_inv
    
    def get_mat_Sio_norm(self, k: float):
        pass
    
    def run(self):
        self._mat_A.fill(0)
        self._mat_P.fill(0)

        self.rhs_constructor.set_boundaries(sqrt_ni=self._sqrt_ni, sqrt_no=self._sqrt_no)
        if self._is_transmission > 0:
            # no scope
            self.matrix_layer_forward(k=self._k, sqrt_n=self._sqrt_no)
            self.assemble_matA(assemble_type=int(AssembleType.ADD_P_MINUS), multiplier=1.0)  # Add P_MINUS_O
            self.assemble_matP(assemble_type=int(AssembleType.ADD_P_MINUS), multiplier=1.0)  # ADD P_MILUS_O
            self.assemble_rhs()  # 0.5I - M_O
            # ni scope
            self.matrix_layer_forward(k=self._k, sqrt_n=self._sqrt_ni)
            self.assemble_matA(assemble_type=int(AssembleType.ADD_P_PLUS), multiplier=-1.0)  # Reduce P_PLUS_I
            self._mat_P.fill(0)
            self.assemble_matP(assemble_type=int(AssembleType.ADD_P_PLUS), multiplier=1.0)  # ADD P_PLUS_I
        else:
            self.matrix_layer_forward(k=self._k)
            self.assemble_matA(assemble_type=int(AssembleType.ADD_M), multiplier=-1.0)  # Reduce M
            self.rhs_layer_forward(assemble_type=int(AssembleType.ADD_P_PLUS), k=self._k)
            self.assemble_rhs()  # 0.5I + M

        # Solve
        self.solve()
        
        # Post process
        self.compute_analytical_raw_solved()
        self.splat_u_from_raw_solved_to_vertices()

        # Prepare for draw your color on GUI
        self.compute_color()
        self.set_mesh_instances()
    
    def solve(self):
        # residual = self.Jacobian_solver()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_mat_A = warp_tensor(self._mat_A.to_torch().to(device))
        torch_rhs = warp_tensor(self.rhs.to_torch().to(device))
        
        torch_mat = torch.zeros((self.rhs.shape[0], self._mat_A.shape[1])).to(torch_mat_A.dtype).to(device)
        torch_mat[0: (self.N_Neumann + self.M_Dirichlet), ...] = torch_mat_A
        if self._is_transmission > 0:
            torch_mat_P = warp_tensor(self._mat_P.to_torch().to(device))
            torch_mat[(self.N_Neumann + self.M_Dirichlet) :, ...] = torch_mat_P
            
            torch_solved, _, _, _ = torch.linalg.lstsq(torch_mat, torch_rhs.unsqueeze(-1))
            torch_solved = torch_solved.squeeze(-1)
        else:
            torch_solved = torch.linalg.solve(torch_mat, torch_rhs)
        # if ti.static(self._is_transmission > 0):
        #     self._mat_P.fill(0)
        #     self.matrix_layer_forward(k=self._k, sqrt_n=self._sqrt_no)
        #     self.assemble_matP(assemble_type=int(AssembleType.ADD_P_MINUS), multiplier=1.0)  # ADD P_MINUS_O
        #     torch_mat_P = warp_tensor(self._mat_P.to_torch().to(device))

        #     torch_solved_norm = torch_solved.norm()
        #     print("solved_norm = ", torch_solved_norm)

        #     torch_solved = torch.matmul(torch_mat_P, torch_solved)

        #     torch_solved_norm = torch_solved.norm()
        #     print("solved_norm = ", torch_solved_norm)
        
        torch_solved = unwarp_tensor(torch_solved)

        self.raw_solved.from_torch(torch_solved)

    def kill(self):
        self.initialized = False

        self.num_of_vertices = 0
        self.num_of_panels = 0

        self.solved = None
        self.analytical_solved = None

        self.raw_solved = None
        self.raw_analytical_solved = None
        self._mat_A = None
        self.rhs = None
        self._mat_P = None

        # For visualization
        self.solved_vert_color = None
        self.solved_vertices = None
        self.analytical_vert_color = None
        self.analytical_vertices = None
        self.diff_vert_color = None
        self.diff_vertices = None

        # For numerical integrations
        self.m_mats_coincide = None
        self.Gauss_points_1d = None
        self.Gauss_weights_1d = None

        self.single_layer.kill()
        self.double_layer.kill()
        self.adj_double_layer.kill()
        self.hypersingular_layer.kill()
        self.identity_layer.kill()
        self.rhs_constructor.kill()
        self._log_manager.ErrorLog("Kill the BEM Manager")
