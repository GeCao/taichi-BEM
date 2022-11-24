import os
import numpy as np
import taichi as ti
import torch

from src.BEM_functions.single_layer import SingleLayer2d, SingleLayer3d
from src.BEM_functions.double_layer import DoubleLayer2d, DoubleLayer3d
from src.BEM_functions.adj_double_layer import AdjDoubleLayer2d, AdjDoubleLayer3d
from src.BEM_functions.hypersingular_layer import HypersingularLayer2d, HypersingularLayer3d
from src.BEM_functions.rhs_constructor import RHSConstructor3d
from src.managers.mesh_manager import KernelType


@ti.data_oriented
class BEMManager:
    def __init__(self, core_manager):
        self._core_manager = core_manager
        self._log_manager = core_manager._log_manager

        self._np_dtype = self._core_manager._np_dtype
        self._ti_dtype = self._core_manager._ti_dtype
        self._kernel_type = int(KernelType.LAPLACE)

        self._simulation_parameters = self._core_manager._simulation_parameters
        if self._simulation_parameters["kernel"] == "Laplace":
            self._kernel_type = int(KernelType.LAPLACE)
            self._n = 1
        elif self._simulation_parameters["kernel"] == "Helmholtz":
            self._kernel_type = int(KernelType.HELMHOLTZ)
            self._n = 2
        else:
            raise RuntimeError("Kernel Type only support Laplace and Helmholtz for now")
        
        self._k = self._simulation_parameters["k"]
        
        self.num_of_vertices = 0
        self.num_of_panels = 0

        self._dim = self._simulation_parameters['dim']
        self._Q = self._dim
        self._GaussQR = self._simulation_parameters['GaussQR']

        self.initialized = False

    def initialization(self, analyical_function_Dirichlet, analyical_function_Neumann):
        if not self._core_manager._log_manager.initialized:
            raise RuntimeError("The initialization of Log Manager has the first priority than others!")
        
        if not self._core_manager._mesh_manager.initialized:
            raise RuntimeError("The initialization of Mesh Manager should always before the BEM Manager")
        
        self.num_of_vertices = self._core_manager._mesh_manager.get_num_of_vertices()
        self.num_of_panels = self._core_manager._mesh_manager.get_num_of_panels()
        self.num_of_Dirichlets = self._core_manager._mesh_manager.get_num_of_Dirichlets()
        self.num_of_Neumanns = self._core_manager._mesh_manager.get_num_of_Neumanns()

        self.analyical_function_Dirichlet = analyical_function_Dirichlet
        self.analyical_function_Neumann = analyical_function_Neumann

        self.solved = ti.Vector.field(self._n, self._ti_dtype, shape=(self.num_of_vertices, ))
        self.analytical_solved = ti.Vector.field(self._n, self._ti_dtype, shape=(self.num_of_vertices, ))
        self.panels_solved = ti.Vector.field(self._n, self._ti_dtype, shape=(self.num_of_panels * self._dim, ))
        self.mat_A = ti.Vector.field(self._n, self._ti_dtype, shape=(self.num_of_panels * self._dim, self.num_of_panels * self._dim))
        self.rhs = ti.Vector.field(self._n, self._ti_dtype, shape=(self.num_of_panels * self._dim, ))

        self.solved_vert_color = ti.Vector.field(self._dim, self._ti_dtype, shape=(self.num_of_vertices, ))
        self.solved_vertices = ti.Vector.field(self._dim, self._ti_dtype, shape=(self.num_of_vertices, ))
        self.analytical_vert_color = ti.Vector.field(self._dim, self._ti_dtype, shape=(self.num_of_vertices, ))
        self.analytical_vertices = ti.Vector.field(self._dim, self._ti_dtype, shape=(self.num_of_vertices, ))
        self.default_ones = ti.field(dtype=self._ti_dtype, shape=(self.num_of_vertices))
        np_default_ones = np.array([1.0 for i in range(self.num_of_vertices)], dtype=self._np_dtype)
        self.default_ones.from_numpy(np_default_ones)
        if self._dim == 2:
            self.single_layer = SingleLayer2d(self)
            self.double_layer = DoubleLayer2d(self)
            self.adj_double_layer = AdjDoubleLayer2d(self)
            self.hypersingular_layer = HypersingularLayer2d(self)
        elif self._dim == 3:
            self.single_layer = SingleLayer3d(self)
            self.double_layer = DoubleLayer3d(self)
            self.adj_double_layer = AdjDoubleLayer3d(self)
            self.hypersingular_layer = HypersingularLayer3d(self)

            self.rhs_constructor = RHSConstructor3d(self)
        else:
            raise RuntimeError("The dimension of our scene should only be 2D or 3D")

        self.initialized = True
    
    def get_num_of_vertices(self):
        return self.num_of_vertices
    
    def get_num_of_panels(self):
        return self.num_of_panels

    def get_num_of_Dirichlets(self):
        return self.num_of_Dirichlets
    
    def get_num_of_Neumanns(self):
        return self.num_of_Neumanns

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
        return self._core_manager._mesh_manager.normals[vert_index]
    
    @ti.func
    def map_local_Dirichlet_index_to_panel_index(self, local_index):
        return self._core_manager._mesh_manager.map_local_Dirichlet_index_to_panel_index(local_index)
    
    @ti.func
    def map_local_Neumann_index_to_panel_index(self, local_index):
        return self._core_manager._mesh_manager.map_local_Neumann_index_to_panel_index(local_index)
    
    @ti.func
    def get_panels_relation(self, i, j):
        return self._core_manager._mesh_manager.get_panels_relation(i, j)
    
    @ti.func
    def get_panel_type(self, panel_index):
        return self._core_manager._mesh_manager.get_panel_types()[panel_index]
    
    @ti.kernel
    def compute_color(self):
        max_solved = -66666.0
        for I in self.analytical_solved:
            if self.analytical_solved[I].x > 0:
                self.analytical_vert_color[I].x = self.analytical_solved[I].norm()
                if max_solved < self.analytical_solved[I].norm():
                    max_solved = self.analytical_solved[I].norm()
            else:
                self.analytical_vert_color[I].z = self.analytical_solved[I].norm()
                if max_solved < self.analytical_solved[I].norm():
                    max_solved = self.analytical_solved[I].norm()
        
        for I in self.solved:
            if self.solved[I].x > 0:
                self.solved_vert_color[I].x = self.solved[I].norm()
                if max_solved < self.solved[I].norm():
                    max_solved = self.solved[I].norm()
            else:
                self.solved_vert_color[I].z = self.solved[I].norm()
                if max_solved < self.solved[I].norm():
                    max_solved = self.solved[I].norm()
        
        for I in self.solved_vert_color:
            self.solved_vert_color[I] /= max_solved
        
        for I in self.analytical_vert_color:
            self.analytical_vert_color[I] /= max_solved
        
        for I in self.solved_vertices:
            self.analytical_vertices[I] = self.get_vertice(I)
            self.analytical_vertices[I].z += 1.8

            self.solved_vertices[I] = self.get_vertice(I)
            self.solved_vertices[I].z -= 1.8
            
    @ti.kernel
    def splat_u_from_panels_to_vertices(self):
        for I in self.solved:
            self.solved[I] = 0 * self.solved[I]
        
        if ti.static(self._dim == 3):
            GaussQR2 = self._GaussQR * self._GaussQR       
            for I in self.panels_solved:
                i = I // self._dim
                ii = I % self._dim
                x1 = self._core_manager._mesh_manager.vertices[self._core_manager._mesh_manager.panels[i * self._dim + 0]]
                x2 = self._core_manager._mesh_manager.vertices[self._core_manager._mesh_manager.panels[i * self._dim + 1]]
                x3 = self._core_manager._mesh_manager.vertices[self._core_manager._mesh_manager.panels[i * self._dim + 2]]
                area = self._core_manager._mesh_manager.get_panel_areas()[i]
                for iii in range(GaussQR2):
                    r1 = self.rhs_constructor.Gauss_points_1d[iii // self._GaussQR]
                    r2 = self.rhs_constructor.Gauss_points_1d[iii % self._GaussQR] * r1

                    weight = self.rhs_constructor.Gauss_weights_1d[iii // self._GaussQR] * self.rhs_constructor.Gauss_weights_1d[iii % self._GaussQR] * (area * 2.0)

                    x = self.rhs_constructor.interplate_from_unit_triangle_to_general(
                        r1=r1, r2=r2, x1=x1, x2=x2, x3=x3
                    )
                    vert_index = self._core_manager._mesh_manager.panels[I]
                    jacobian = r1

                    self.solved[vert_index] += (
                        self.panels_solved[I] * self.rhs_constructor.shape_function(r1, r2, i=ii)
                    ) * weight * jacobian
            
            for I in self.solved:
                self.solved[I] /= self._core_manager._mesh_manager.get_vert_areas()[I]

    @ti.kernel
    def assmeble(self):
        for I, J in self.mat_A:
            self.mat_A[I, J] = self.single_layer._Vmat[I, J]
        
        if ti.static(self.rhs_constructor.num_of_Dirichlets > 0):
            for I in range(self.rhs_constructor.num_of_Dirichlets * self.rhs_constructor._Q):
                self.rhs[I] = self.rhs_constructor._gvec[I]
                x = self._core_manager._mesh_manager.vertices[I]
                normal_x = self._core_manager._mesh_manager.get_vert_normals()[I]
                self.analytical_solved[I] = self.rhs_constructor.analyical_function_Neumann(x, normal_x)
        
        offset = self.rhs_constructor.num_of_Dirichlets * self.rhs_constructor._Q
        if ti.static(self.rhs_constructor.num_of_Neumanns > 0):
            for I in range(self.rhs_constructor.num_of_Neumanns * self.rhs_constructor._Q):
                self.rhs[I + offset] = self.rhs_constructor._fvec[I]
                x = self._core_manager._mesh_manager.vertices[I]
                self.analytical_solved[I + offset] = self.rhs_constructor.analyical_function_Dirichlet(x)

    @ti.kernel
    def Jacobian_solver(self) -> float:
        residual = 0.0
        for I in self.panels_solved:
            new_u = self.rhs[I] + self.mat_A[I, I] * self.panels_solved[I]
            for J in range(self.panels_solved.shape[0]):
                new_u -= self.mat_A[I, J] * self.panels_solved[J]
            new_u /= self.mat_A[I, I]
            # residual += (self.panels_solved[I] - new_u) * (self.panels_solved[I] - new_u)
            self.panels_solved[I] = new_u
        
        for I in self.panels_solved:
            b_Ax = self.rhs[I]
            for J in range(self.panels_solved.shape[0]):
                b_Ax -= self.mat_A[I, J] * self.panels_solved[J]
            residual += b_Ax * b_Ax
        
        residual = ti.math.sqrt(residual / self.panels_solved.shape[0])
        return residual

    def run(self):
        self._log_manager.InfoLog("Construct Single Layer")
        self.single_layer.forward()
        self._log_manager.InfoLog("Construct Double Layer")
        self.double_layer.forward()
        self._log_manager.InfoLog("Construct HyperSingular Layer")
        self.hypersingular_layer.forward()
        self._log_manager.InfoLog("Construct RHS Layer")
        self.rhs_constructor.forward()

        self._log_manager.InfoLog("Construct all Layers Done")
        self.assmeble()

        # Solve
        np_mat_A = self.mat_A.to_numpy()
        np_rhs = self.rhs.to_numpy()
        if np_mat_A.shape[-1] == 1:
            np_mat_A = np_mat_A.squeeze(-1)
        elif np_mat_A.shape[-1] == 2:
            np_mat_A = np_mat_A[..., 0] + np_mat_A[..., 1] * 1j
        if np_rhs.shape[-1] == 1:
            np_rhs.squeeze(-1)
        elif np_rhs.shape[-1] == 2:
            np_rhs = np_rhs[..., 0] + np_rhs[..., 1] * 1j
        
        device = torch.device("cuda")
        torch_mat_A = torch.from_numpy(np_mat_A).to(device)
        torch_rhs = torch.from_numpy(np_rhs).to(device)
        torch_solved = torch.linalg.solve(torch_mat_A, torch_rhs)
        np_solved = torch_solved.cpu().numpy()
        if self._n == 1:
            np_solved = np.expand_dims(np_solved, axis=-1)
        elif self._n == 2:
            np_solved = np.stack(
                (np.real(np_solved), np.imag(np_solved)),
                axis=-1
            )

        self.panels_solved.from_numpy(np_solved)
        print("solve min = {}, max = {}, mean = {}".format(
            np.min(np.linalg.norm(np_solved, axis=-1)), np.max(np.linalg.norm(np_solved, axis=-1)), np.mean(np.linalg.norm(np_solved, axis=-1)))
        )
        np_analytical_solved = self.analytical_solved.to_numpy()
        print("analytical sovle min = {}, max = {}, mean = {}".format(
            np.min(np.linalg.norm(np_analytical_solved, axis=-1)), np.max(np.linalg.norm(np_analytical_solved, axis=-1)), np.mean(np.linalg.norm(np_analytical_solved, axis=-1)))
        )
        residual = np.mean(np.abs(np_solved))
        if self._core_manager.iteration <= 15:
            self._log_manager.InfoLog("residual = {}".format(residual))
        self.splat_u_from_panels_to_vertices()
        self.compute_color()
    
    def solve(self):
        residual = self.Jacobian_solver()
        if self._core_manager.iteration <= 15:
            self._log_manager.InfoLog("residual = {}".format(residual))
        self.splat_u_from_panels_to_vertices()
        self.compute_color()

    def kill(self):
        self.initialized = False

        self.num_of_vertices = 0
        self.num_of_panels = 0

        self.single_layer.kill()
        self.double_layer.kill()
        self.adj_double_layer.kill()
        self.hypersingular_layer.kill()
        self._log_manager.ErrorLog("Kill the BEM Manager")
