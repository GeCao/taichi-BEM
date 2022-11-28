import os
import numpy as np
import math
import taichi as ti
import torch

from src.BEM_functions.single_layer import SingleLayer2d, SingleLayer3d
from src.BEM_functions.double_layer import DoubleLayer2d, DoubleLayer3d
from src.BEM_functions.adj_double_layer import AdjDoubleLayer2d, AdjDoubleLayer3d
from src.BEM_functions.hypersingular_layer import HypersingularLayer2d, HypersingularLayer3d
from src.BEM_functions.rhs_constructor import RHSConstructor3d
from src.managers.mesh_manager import KernelType, CellFluxType, VertAttachType
from src.BEM_functions.utils import get_gaussion_integration_points_and_weights


@ti.data_oriented
class BEMManager:
    def __init__(self, core_manager):
        self._core_manager = core_manager
        self._log_manager = core_manager._log_manager

        self._np_dtype = self._core_manager._np_dtype
        self._ti_dtype = self._core_manager._ti_dtype
        self._kernel_type = int(KernelType.LAPLACE)
        self._boundary = int(CellFluxType.TOBESOLVED)

        self._simulation_parameters = self._core_manager._simulation_parameters
        if self._simulation_parameters["kernel"] == "Laplace":
            self._kernel_type = int(KernelType.LAPLACE)
            self._n = 1
        elif self._simulation_parameters["kernel"] == "Helmholtz":
            self._kernel_type = int(KernelType.HELMHOLTZ)
            self._n = 2
        else:
            raise RuntimeError("Kernel Type only support Laplace and Helmholtz for now")
        
        if self._simulation_parameters["boundary"] == "Dirichlet":
            self._boundary = int(CellFluxType.TOBESOLVED)
        elif self._simulation_parameters["boundary"] == "Neumann":
            self._boundary = int(CellFluxType.NEUMANN_KNOWN)
        elif self._simulation_parameters["boundary"] == "Mix":
            self._boundary = int(CellFluxType.MIX)
        else:
            raise RuntimeError("Boundary Type only support DIrichlet/Neumann/Mix for now")
        
        self._k = self._simulation_parameters["k"]
        self._sqrt_ni = math.sqrt(self._simulation_parameters["n_i"])
        self._sqrt_no = math.sqrt(self._simulation_parameters["n_o"])
        self._use_augment = int(self._simulation_parameters["use_augment"])
        
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

        # For final comparision
        self.solved = ti.Vector.field(self._n, self._ti_dtype, shape=(self.num_of_vertices, ))
        self.analytical_solved = ti.Vector.field(self._n, self._ti_dtype, shape=(self.num_of_vertices, ))

        # The final global Ax=rhs
        self.raw_solved = ti.Vector.field(self._n, self._ti_dtype, shape=(self.num_of_Dirichlets + self.num_of_Neumanns, ))
        self.mat_A = ti.Vector.field(
            self._n,
            self._ti_dtype,
            shape=(self.num_of_Dirichlets + self.num_of_Neumanns, self.num_of_Dirichlets + self.num_of_Neumanns)
        )
        self.rhs = ti.Vector.field(self._n, self._ti_dtype, shape=(self.num_of_Dirichlets + self.num_of_Neumanns, ))
        self._mat_P = ti.Vector.field(self._n, self._ti_dtype, shape=())
        if self._use_augment > 0:
            self.rhs = ti.Vector.field(self._n, self._ti_dtype, shape=(2 * (self.num_of_Dirichlets + self.num_of_Neumanns), ))
            self._mat_P = ti.Vector.field(
                self._n,
                self._ti_dtype,
                shape=(self.num_of_Dirichlets + self.num_of_Neumanns, self.num_of_Dirichlets + self.num_of_Neumanns)
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
    def shape_function(self, r1, r2, i: int):
        """
        Providing x (r1, r2) = (1 - r1) * x1 + (r1 - r2) * x2 + r2 * x3
        """
        result = 1.0  # Default as 1 to cancel shape function term
        if i == 0:
            result = 1 - r1
        elif i == 1:
            result = r1 - r2
        elif i == 2:
            result = r2
        return result
    
    @ti.func
    def G(self, x, y):
        Gxy = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
        distance = (x - y).norm()
        if ti.static(self._kernel_type == int(KernelType.LAPLACE)):
            Gxy.x = (1.0 / 4.0 / ti.math.pi) / distance
        elif ti.static(self._kernel_type == int(KernelType.HELMHOLTZ)):
            Gxy = (1.0 / 4.0 / ti.math.pi) / distance * ti.math.cexp(
                ti.Vector([0.0, self._k * distance], self._ti_dtype)
            )
        
        return Gxy
    
    @ti.func
    def grad_G_y(self, x, y, normal_y):
        grad_Gy = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
        distance = (x - y).norm()
        if ti.static(self._kernel_type == int(KernelType.LAPLACE)):
            grad_Gy.x = (1.0 / 4.0 / ti.math.pi) * (x - y).dot(normal_y) / ti.math.pow(distance, 3)
        elif ti.static(self._kernel_type == int(KernelType.HELMHOLTZ)):
            chained_complex_vector = ti.math._complex.cmul(
                ti.Vector([1.0, -self._k * distance], self._ti_dtype),
                ti.math.cexp(ti.Vector([0.0, self._k * distance], self._ti_dtype))
            )
            grad_Gy = (1.0 / 4.0 / ti.math.pi) * (x - y).dot(normal_y) / ti.math.pow(distance, 3) * chained_complex_vector
        
        return grad_Gy
    
    @ti.func
    def grad2_G_xy(self, x, y, curl_phix_dot_curl_phiy):
        result = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
        distance = (x - y).norm()
        if ti.static(self._kernel_type == int(KernelType.LAPLACE)):
            result.x = (1.0 / 4.0 / ti.math.pi) / distance * curl_phix_dot_curl_phiy
        elif ti.static(self._kernel_type == int(KernelType.HELMHOLTZ)):
            exp_vector = ti.math.cexp(
                ti.Vector([0.0, self._k * distance], self._ti_dtype)
            )
            result = (1.0 / 4.0 / ti.math.pi) / distance * curl_phix_dot_curl_phiy * exp_vector
        return result
    
    @ti.kernel
    def compute_color(self):
        # Please Note, if you are applying Neumann boundary, and want to get Dirichlet solve
        # since f'(x) = [f(x) + C]', where C is a constant number,
        # our solved Dirichlet usually does not hold the same scales as the analytical solves
        # In this case, scale our Dirichlet solve to the analytical Dirichlets

        max_analytical_Neumann_boundary = -5e11
        max_analytical_Dirichlet_boundary = -5e11
        mean_solved_Neumann_boundary = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
        mean_solved_Dirichlet_boundary = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
        mean_analytical_Neumann_boundary = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
        mean_analytical_Dirichlet_boundary = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)

        for i in range(self.num_of_vertices):
            if self.get_vertice_type(i) == int(VertAttachType.DIRICHLET_KNOWN):
                # Dirichlet boundary, solved variables is Neumann
                mean_solved_Dirichlet_boundary += self.solved[i] / (self.num_of_panels - self.num_of_Neumanns)
                mean_analytical_Dirichlet_boundary += self.analytical_solved[i] / (self.num_of_panels - self.num_of_Neumanns)
                ti.atomic_max(max_analytical_Dirichlet_boundary, self.analytical_solved[i].norm())
            else:
                # Neumann boundary, solved variables is Dirichlet
                mean_solved_Neumann_boundary += self.solved[i] / self.num_of_Neumanns
                mean_analytical_Neumann_boundary += self.analytical_solved[i] / self.num_of_Neumanns
                ti.atomic_max(max_analytical_Neumann_boundary, self.analytical_solved[i].norm())
        
        # Do your scale:
        for i in range(self.num_of_vertices):
            if self.get_vertice_type(i) == int(VertAttachType.TOBESOLVED):
                # Neumann boundary, solved variables is Dirichlet
                self.solved[i] += mean_analytical_Neumann_boundary - mean_solved_Neumann_boundary
                
        for i in range(self.num_of_vertices):
            if self.get_vertice_type(i) == int(VertAttachType.DIRICHLET_KNOWN):
                # Dirichlet boundary, solved variables is Neumann
                if self.analytical_solved[i].x > 0:
                    self.analytical_vert_color[i].x = self.analytical_solved[i].norm() / max_analytical_Dirichlet_boundary
                else:
                    self.analytical_vert_color[i].z = self.analytical_solved[i].norm() / max_analytical_Dirichlet_boundary
                
                if self.solved[i].x > 0:
                    self.solved_vert_color[i].x = self.solved[i].norm() / max_analytical_Dirichlet_boundary
                else:
                    self.solved_vert_color[i].z = self.solved[i].norm() / max_analytical_Dirichlet_boundary
                
            else:
                # Neumann boundary, solved variables is Dirichlet
                if self.analytical_solved[i].x > 0:
                    self.analytical_vert_color[i].x = self.analytical_solved[i].norm() / max_analytical_Neumann_boundary
                else:
                    self.analytical_vert_color[i].z = self.analytical_solved[i].norm() / max_analytical_Neumann_boundary
                
                if self.solved[i].x > 0:
                    self.solved_vert_color[i].x = self.solved[i].norm() / max_analytical_Neumann_boundary
                else:
                    self.solved_vert_color[i].z = self.solved[i].norm() / max_analytical_Neumann_boundary
            
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
    def splat_u_from_raw_solved_to_vertices(self):
        self.solved.fill(0)
        
        if ti.static(self._dim == 3):
            Dirichlet_boundary_offset = 0
            for I in range(self.num_of_Dirichlets):
                # Dirichlet boundary
                global_i = self.map_local_Dirichlet_index_to_panel_index(I)

                x1_idx = self.get_vertice_index_from_flat_panel_index(self._dim * global_i + 0)
                x2_idx = self.get_vertice_index_from_flat_panel_index(self._dim * global_i + 1)
                x3_idx = self.get_vertice_index_from_flat_panel_index(self._dim * global_i + 2)

                self.solved[x1_idx] += self.raw_solved[Dirichlet_boundary_offset + I] * self.get_panel_area(global_i) / 3.0 / self.get_vert_area(x1_idx)
                self.solved[x2_idx] += self.raw_solved[Dirichlet_boundary_offset + I] * self.get_panel_area(global_i) / 3.0 / self.get_vert_area(x2_idx)
                self.solved[x3_idx] += self.raw_solved[Dirichlet_boundary_offset + I] * self.get_panel_area(global_i) / 3.0 / self.get_vert_area(x3_idx)
            
            Neumann_boundary_offset = self.num_of_Dirichlets
            if ti.static(self.num_of_vertices > 0):
                for global_i in range(self.num_of_vertices):
                    if self.get_vertice_type(global_i) == int(VertAttachType.TOBESOLVED):
                        local_I = self.map_global_vert_index_to_local_Neumann(global_i)
                        self.solved[global_i] = self.raw_solved[Neumann_boundary_offset + local_I]

    @ti.kernel
    def assmeble(self):
        if ti.static(self._use_augment > 0):
            pass
        else:
            if ti.static(self.num_of_Dirichlets > 0):
                for I, J in self.single_layer._Vmat:
                    self.mat_A[I, J] = self.single_layer._Vmat[I, J]
            
            if ti.static(self.num_of_Dirichlets * self.num_of_Neumanns > 0):
                for I, J in self.double_layer._Kmat:
                    self.mat_A[I, J + self.num_of_Dirichlets] = -self.double_layer._Kmat[I, J]
            
            if ti.static(self.num_of_Dirichlets * self.num_of_Neumanns > 0):
                for I, J in self.adj_double_layer._Kmat:
                    self.mat_A[I + self.num_of_Dirichlets, J] = self.adj_double_layer._Kmat[I, J]
            
            if ti.static(self.num_of_Neumanns > 0):
                for I, J in self.hypersingular_layer._Wmat:
                    self.mat_A[I + self.num_of_Dirichlets, J + self.num_of_Dirichlets] = self.hypersingular_layer._Wmat[I, J]
            
            if ti.static(self.num_of_Dirichlets > 0):
                for I in self.rhs_constructor._gvec:
                    self.rhs[I] = self.rhs_constructor._gvec[I]
            
            if ti.static(self.num_of_Neumanns > 0):
                for I in self.rhs_constructor._fvec:
                    self.rhs[I + self.num_of_Dirichlets] = self.rhs_constructor._fvec[I]
        
        for I in range(self.num_of_panels):
            for ii in range(self._dim):
                # Dirichlet boundary
                if self.get_panel_type(I) == int(CellFluxType.TOBESOLVED):
                    vert_index = self.get_vertice_index_from_flat_panel_index(self._dim * I + ii)
                    x = self.get_vertice(vert_index)
                    normal_x = self.get_vert_normal(vert_index)
                    fx = self.rhs_constructor.analyical_function_Neumann(x, normal_x)
                    self.analytical_solved[vert_index].x = fx.x
        
        for I in range(self.num_of_panels):
            for ii in range(self._dim):
                # Neumann boundary
                if self.get_panel_type(I) == int(CellFluxType.NEUMANN_KNOWN):
                    vert_index = self.get_vertice_index_from_flat_panel_index(self._dim * I + ii)
                    x = self.get_vertice(vert_index)
                    gx = self.rhs_constructor.analyical_function_Dirichlet(x)
                    self.analytical_solved[vert_index].x = gx.x

    @ti.kernel
    def Jacobian_solver(self) -> float:
        residual = 0.0
        for I in self.raw_solved:
            new_u = self.rhs[I] + self.mat_A[I, I] * self.raw_solved[I]
            for J in range(self.raw_solved.shape[0]):
                new_u -= self.mat_A[I, J] * self.raw_solved[J]
            new_u /= self.mat_A[I, I]
            # residual += (self.raw_solved[I] - new_u) * (self.raw_solved[I] - new_u)
            self.raw_solved[I] = new_u
        
        for I in self.raw_solved:
            b_Ax = self.rhs[I]
            for J in range(self.raw_solved.shape[0]):
                b_Ax -= self.mat_A[I, J] * self.raw_solved[J]
            residual += b_Ax * b_Ax
        
        residual = ti.math.sqrt(residual / self.raw_solved.shape[0])
        return residual

    def run(self):
        self._log_manager.InfoLog("Construct Single Layer")
        self.single_layer.forward()
        self._log_manager.InfoLog("Construct Double Layer")
        self.double_layer.forward()
        self._log_manager.InfoLog("Construct Adj Double Layer")
        self.adj_double_layer.forward()
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
            np_solved[..., 1] = 0

        self.raw_solved.from_numpy(np_solved)

        self.splat_u_from_raw_solved_to_vertices()

        self.compute_color()
        self.set_mesh_instances()
    
    def solve(self):
        residual = self.Jacobian_solver()
        if self._core_manager.iteration <= 15:
            self._log_manager.InfoLog("residual = {}".format(residual))
        self.splat_u_from_raw_solved_to_vertices()
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
