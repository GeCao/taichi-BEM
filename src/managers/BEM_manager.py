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
        elif self._simulation_parameters["kernel"] == "Helmholtz":
            self._kernel_type = int(KernelType.HELMHOLTZ)
        else:
            raise RuntimeError("Kernel Type only support Laplace and Helmholtz for now")
        
        self._k = self._simulation_parameters["k"]
        
        self.num_of_vertices = 0
        self.num_of_panels = 0

        self.dim = self._simulation_parameters['dim']
        self._GaussQR = 7

        self.initialized = False

    def initialization(self):
        if not self._core_manager._log_manager.initialized:
            raise RuntimeError("The initialization of Log Manager has the first priority than others!")
        
        if not self._core_manager._mesh_manager.initialized:
            raise RuntimeError("The initialization of Mesh Manager should always before the BEM Manager")
        
        self.num_of_vertices = self._core_manager._mesh_manager.num_of_vertices
        self.num_of_panels = self._core_manager._mesh_manager.num_of_panels

        self.solved = ti.field(self._ti_dtype, shape=(self.num_of_vertices, ))
        self.analytical_solved = ti.field(self._ti_dtype, shape=(self.num_of_vertices, ))
        self.panels_solved = ti.field(self._ti_dtype, shape=(self.num_of_panels * self.dim, ))
        self.rhs_computed = ti.field(self._ti_dtype, shape=(self.num_of_panels * self.dim, ))
        self.mat_A = ti.field(self._ti_dtype, shape=(self.num_of_panels * self.dim, self.num_of_panels * self.dim))
        self.rhs = ti.field(self._ti_dtype, shape=(self.num_of_panels * self.dim, ))

        self.vert_color = ti.Vector.field(self.dim, self._ti_dtype, shape=(self.num_of_vertices, ))
        self.default_ones = ti.field(
            dtype=self._ti_dtype,
            shape=(self.num_of_vertices)
        )
        np_default_ones = np.array(
            [1.0 for i in range(self.num_of_vertices)], dtype=self._np_dtype
        )
        self.default_ones.from_numpy(np_default_ones)
        if self.dim == 2:
            self.single_layer = SingleLayer2d(self, self._core_manager._mesh_manager)
            self.double_layer = DoubleLayer2d(self, self._core_manager._mesh_manager)
            self.adj_double_layer = AdjDoubleLayer2d(self)
            self.hypersingular_layer = HypersingularLayer2d(self, self._core_manager._mesh_manager)
        elif self.dim == 3:
            self.single_layer = SingleLayer3d(self, self._core_manager._mesh_manager)
            self.double_layer = DoubleLayer3d(self, self._core_manager._mesh_manager)
            self.adj_double_layer = AdjDoubleLayer3d(self)
            self.hypersingular_layer = HypersingularLayer3d(self, self._core_manager._mesh_manager)

            self.rhs_constructor = RHSConstructor3d(self, self._core_manager._mesh_manager)
        else:
            raise RuntimeError("The dimension of our scene should only be 2D or 3D")

        self.initialized = True
    
    @ti.kernel
    def compute_color(self):
        max_solved = -66666.0
        for I in self.solved:
            if self.solved[I] > 0:
                self.vert_color[I].x = self.solved[I]
                if max_solved < self.solved[I]:
                    max_solved = self.solved[I]
            else:
                self.vert_color[I].z = -self.solved[I]
                if max_solved < -self.solved[I]:
                    max_solved = self.solved[I]
        
        for I in self.vert_color:
            self.vert_color[I].x /= max_solved
            self.vert_color[I].y /= max_solved
            self.vert_color[I].z /= max_solved
            
    @ti.kernel
    def splat_u_from_panels_to_vertices(self):
        for I in self.solved:
            self.solved[I] = 0.0
        
        if ti.static(self.dim == 3):
            GaussQR2 = self._GaussQR * self._GaussQR       
            for I in self.panels_solved:
                i = I // self.dim
                ii = I % self.dim
                x1 = self._core_manager._mesh_manager.vertices[self._core_manager._mesh_manager.panels[i * self.dim + 0]]
                x2 = self._core_manager._mesh_manager.vertices[self._core_manager._mesh_manager.panels[i * self.dim + 1]]
                x3 = self._core_manager._mesh_manager.vertices[self._core_manager._mesh_manager.panels[i * self.dim + 2]]
                area = self._core_manager._mesh_manager.get_panel_areas()[i]
                for iii in range(GaussQR2):
                    r1 = self.rhs_constructor.Gauss_points_1d[iii // self._GaussQR]
                    r2 = self.rhs_constructor.Gauss_points_1d[iii % self._GaussQR] * r1

                    weight = self.rhs_constructor.Gauss_weights_1d[iii // self._GaussQR] * self.rhs_constructor.Gauss_weights_1d[iii % self._GaussQR] * area * 2.0

                    x = self.rhs_constructor.interplate_from_unit_triangle_to_general(
                        r1=r1, r2=r2, x1=x1, x2=x2, x3=x3
                    )
                    vert_index = self._core_manager._mesh_manager.panels[I]
                    jacobian = r1

                    self.solved[vert_index] += self.panels_solved[I] * self.rhs_constructor.shape_function(r1, r2, i=ii) * weight * jacobian
            
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
            self.rhs_computed[I] = b_Ax
        
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
        device = torch.device("cuda")
        torch_mat_A = torch.from_numpy(np_mat_A).to(device)
        torch_rhs = torch.from_numpy(np_rhs).to(device)
        torch_solved = torch.linalg.solve(torch_mat_A, torch_rhs)
        np_solved = torch_solved.cpu().numpy()
        self.panels_solved.from_numpy(np_solved)
        print("solve min = {}, max = {}, mean = {}".format(np.min(np_solved), np.max(np_solved), np.mean(np_solved)))
        np_analytical_solved = self.analytical_solved.to_numpy()
        print("analytical sovle min = {}, max = {}, mean = {}".format(np.min(np_analytical_solved), np.max(np_analytical_solved), np.mean(np_analytical_solved)))
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
