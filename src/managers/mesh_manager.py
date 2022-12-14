import os
import numpy as np
import math
import taichi as ti
import pywavefront

from src.BEM_functions.utils import CellFluxType, VertAttachType, KernelType, PanelsRelation


@ti.data_oriented
class MeshManager:
    def __init__(self, core_manager):
        self._core_manager = core_manager
        self._log_manager = core_manager._log_manager

        self._np_dtype = self._core_manager._np_dtype
        self._ti_dtype = self._core_manager._ti_dtype
        self._dim = self._core_manager._simulation_parameters['dim']
        self._boundary_type = self._core_manager._simulation_parameters['boundary']
        self._is_transmission = self._core_manager._is_transmission

        self._asset_path = os.path.join(self._core_manager.root_path, "assets")
        self._object_path = os.path.join(
            self._asset_path,
            self._core_manager._simulation_parameters['object'] + '.obj'
        )
        self.vertices = None
        self.panels = None

        self.panel_areas = None
        self.vert_areas = None
        self.panel_normals = None
        self.vert_normals = None

        self.panel_types = None
        self.vertice_types = None

        self.Dirichlet_index = None
        self.Neumann_index = None

        self.num_of_vertices = 0
        self.num_of_panels = 0

        self.num_of_panels_Dirichlet = 0
        self.num_of_panels_Neumann = 0

        self.M_Dirichlet = 0
        self.N_Neumann = 0

        self.shape_func_degree_Neumann = 0
        self.shape_func_degree_Dirichlet = 1
        Q_Neumann = self._core_manager._simulation_parameters["Q_Neumann"]
        Q_Dirichlet = self._core_manager._simulation_parameters["Q_Dirichlet"]
        assert(Q_Neumann == 0 or Q_Neumann == 1)
        assert(Q_Dirichlet == 0 or Q_Dirichlet == 1)
        self.shape_func_degree_Neumann = Q_Neumann
        self.shape_func_degree_Dirichlet = Q_Dirichlet

        self.map_global_Neumann_to_local = ti.field(dtype=ti.i32, shape=())
        self.map_global_Dirichlet_to_local = ti.field(dtype=ti.i32, shape=())

        self.Dirichlet_index = ti.field(dtype=ti.i32, shape=())
        self.Neumann_index = ti.field(dtype=ti.i32, shape=())

        self.initialized = False
    
    def initialization(self):
        if not self._core_manager._log_manager.initialized:
            raise RuntimeError("The initialization of Log Manager has the first priority than others!")
        
        self._log_manager.InfoLog("====================================================================")
        self._log_manager.InfoLog("Loading object from path {} started".format(str(self._object_path)))

        if self._dim == 2:
            np_vertices, np_faces = self.load_analytical_disk()
        elif self._dim == 3:
            if "analytical" in self._core_manager._simulation_parameters['object']:
                np_vertices, np_faces = self.load_analytical_sphere()
            else:
                np_vertices, np_faces = self.load_asset(scale=None, translate=None)
        
        np_faces = np_faces.flatten()

        self.vertices = ti.Vector.field(
            n=np_vertices.shape[1], dtype=self._ti_dtype, shape=(np_vertices.shape[0],)
        )  # [NumOfVertices, dim]
        self.panels = ti.field(
            dtype=ti.i32, shape=(np_faces.shape[0],)
        )  # [NumOfFaces * dim]
        self.vertices.from_numpy(np_vertices)
        self.panels.from_numpy(np_faces)
        self.num_of_vertices = np_vertices.shape[0]
        self.num_of_panels = np_faces.shape[0] // self._dim
        self.panel_normals = ti.Vector.field(
            n=self._dim, dtype=self._ti_dtype, shape=(self.num_of_panels,)
        )  # [NUmOfFaces, dim]
        self.vert_normals = ti.Vector.field(
            n=self._dim, dtype=self._ti_dtype, shape=(self.num_of_vertices,)
        )  # [NumOfVertices, dim]
        self.panel_areas = ti.field(dtype=self._ti_dtype, shape=(self.num_of_panels,))  # [NumOfFaces]
        self.vert_areas = ti.field(dtype=self._ti_dtype, shape=(self.num_of_vertices,))  # [NumOfVertices]

        np_panel_areas = np.zeros(self.panel_areas.shape, dtype=self._np_dtype)
        np_vert_areas = np.zeros(self.vert_areas.shape, dtype=self._np_dtype)
        np_panel_normals = np.zeros((self.num_of_panels, self._dim), dtype=self._np_dtype)
        np_vert_normals = np.zeros((self.num_of_vertices, self._dim), dtype=self._np_dtype)

        if self._dim == 2:
            for i in range(self.num_of_panels):
                edge = np_vertices[np_faces[self._dim * i + 1]] - np_vertices[np_faces[self._dim * i + 0]]
                np_panel_areas[i] = np.linalg.norm(edge)
                np_panel_normals[i][0], np_panel_normals[i][1] = edge[1], -edge[0]
                np_panel_normals[i] /= np.linalg.norm(np_panel_normals[i])
        elif self._dim == 3:
            for i in range(self.num_of_panels):
                np_panel_areas[i] = 0.5 * np.linalg.norm(np.cross(
                    np_vertices[np_faces[self._dim * i + 1]] - np_vertices[np_faces[self._dim * i + 0]],
                    np_vertices[np_faces[self._dim * i + 2]] - np_vertices[np_faces[self._dim * i + 1]],
                ))

                np_panel_normals[i] = np.cross(
                    np_vertices[np_faces[self._dim * i + 1]] - np_vertices[np_faces[self._dim * i + 0]],
                    np_vertices[np_faces[self._dim * i + 2]] - np_vertices[np_faces[self._dim * i + 1]],
                )
                np_panel_normals[i] /= np.linalg.norm(np_panel_normals[i])
            
        for i in range(self.num_of_panels * self._dim):
            np_vert_areas[np_faces[i]] += np_panel_areas[i // self._dim] / float(self._dim)
            np_vert_normals[np_faces[i]] += np_panel_normals[i // self._dim]
        
        print("area = ", np_panel_areas.sum())
        
        for i in range(self.num_of_vertices):
            np_vert_normals[i] = np_vert_normals[i] / np.linalg.norm(np_vert_normals[i])
            
        self.panel_areas.from_numpy(np_panel_areas)
        self.vert_areas.from_numpy(np_vert_areas)
        self.panel_normals.from_numpy(np_panel_normals)
        self.vert_normals.from_numpy(np_vert_normals)

        self.panel_types = ti.field(dtype=ti.i32, shape=(self.num_of_panels,))
        self.vertice_types = ti.field(dtype=ti.i32, shape=(self.num_of_vertices,))

        self._log_manager.InfoLog("Loading object from path {} finished, "
                                  "with vertices shape = {}, faces shape = {}".format(
            str(self._object_path), np_vertices.shape, np_faces.shape)
        )
        self._log_manager.InfoLog("====================================================================")

        if self._boundary_type == "Full" or self._is_transmission > 0:
            self.set_full_bvp()
        elif self._boundary_type == 'Dirichlet':
            self.set_Dirichlet_bvp()
        elif self._boundary_type == 'Neumann':
            self.set_Neumann_bvp()
        elif self._boundary_type == "Mix":
            self.set_mixed_bvp()
        else:
            raise RuntimeError("The Boundary Type can only be Dirichlet/Neumann/Mix")
        
        self._log_manager.InfoLog("Neumann To be solved = {}".format(self.N_Neumann))
        self._log_manager.InfoLog("Dirichlet To be solved = {}".format(self.M_Dirichlet))

        self.initialized = True

    def set_full_bvp(self):
        assert(self.num_of_panels == self.panel_types.shape[0])
        assert(self.num_of_vertices == self.vertice_types.shape[0])

        if self.shape_func_degree_Neumann == 0:
            self.N_Neumann = self.num_of_panels
        elif self.shape_func_degree_Neumann == 1:
            self.N_Neumann = self.num_of_vertices
        
        if self.shape_func_degree_Dirichlet == 0:
            self.M_Dirichlet = self.num_of_panels
        elif self.shape_func_degree_Dirichlet == 1:
            self.M_Dirichlet = self.num_of_vertices
        
        self.num_of_panels_Dirichlet = self.num_of_panels
        self.num_of_panels_Neumann = self.num_of_panels

        self.panel_types.fill(int(CellFluxType.BOTH_TOBESOLVED))
        self.vertice_types.fill(int(VertAttachType.BOTH_TOBESOLVED))

        np_Dirichlet_index = np.array([i for i in range(self.num_of_panels)], dtype=np.int32)
        self.Dirichlet_index = ti.field(dtype=ti.i32, shape=(self.num_of_panels,))
        self.Dirichlet_index.from_numpy(np_Dirichlet_index)

        np_Neumann_index = np.array([i for i in range(self.num_of_panels)], dtype=np.int32)
        self.Neumann_index = ti.field(dtype=ti.i32, shape=(self.num_of_panels,))
        self.Neumann_index.from_numpy(np_Neumann_index)

        # Use only degree = 1 & Has Neumann to be solved
        if self.shape_func_degree_Neumann == 1:
            np_map_global_Neumann_to_local = np.array([i for i in range(self.num_of_vertices)], dtype=np.int32)
            self.map_global_Neumann_to_local = ti.field(dtype=ti.i32, shape=(self.num_of_vertices,))
            self.map_global_Neumann_to_local.from_numpy(np_map_global_Neumann_to_local)

        # Use only degree = 1 & Has Dirichlet to be solved
        if self.shape_func_degree_Dirichlet == 1:
            np_map_global_Dirichlet_to_local = np.array([i for i in range(self.num_of_vertices)], dtype=np.int32)
            self.map_global_Dirichlet_to_local = ti.field(dtype=ti.i32, shape=(self.num_of_vertices,))
            self.map_global_Dirichlet_to_local.from_numpy(np_map_global_Dirichlet_to_local)
    
    def set_Dirichlet_bvp(self):
        assert(self.num_of_panels == self.panel_types.shape[0])
        assert(self.num_of_vertices == self.vertice_types.shape[0])

        if self.shape_func_degree_Neumann == 0:
            self.N_Neumann = self.num_of_panels
        elif self.shape_func_degree_Neumann == 1:
            self.N_Neumann = self.num_of_vertices
        
        self.M_Dirichlet = 0
        
        self.num_of_panels_Dirichlet = 0
        self.num_of_panels_Neumann = self.num_of_panels

        self.panel_types.fill(int(CellFluxType.NEUMANN_TOBESOLVED))
        self.vertice_types.fill(int(VertAttachType.NEUMANN_TOBESOLVED))

        np_Neumann_index = np.array([i for i in range(self.num_of_panels)], dtype=np.int32)
        self.Neumann_index = ti.field(dtype=ti.i32, shape=(self.num_of_panels,))
        self.Neumann_index.from_numpy(np_Neumann_index)

        self.Dirichlet_index = ti.field(dtype=ti.i32, shape=())

        # Use only degree = 1 & Has Neumann to be solved
        np_map_global_Neumann_to_local = np.array([i for i in range(self.num_of_vertices)], dtype=np.int32)
        self.map_global_Neumann_to_local = ti.field(dtype=ti.i32, shape=(self.num_of_vertices,))
        self.map_global_Neumann_to_local.from_numpy(np_map_global_Neumann_to_local)

        # Use only degree = 1 & Has Dirichlet to be solved
        self.map_global_Dirichlet_to_local = ti.field(dtype=ti.i32, shape=())
    
    def set_Neumann_bvp(self):
        assert(self.num_of_panels == self.panel_types.shape[0])
        assert(self.num_of_vertices == self.vertice_types.shape[0])

        self.N_Neumann = 0
        if self.shape_func_degree_Dirichlet == 0:
            self.M_Dirichlet = self.num_of_panels
        elif self.shape_func_degree_Dirichlet == 1:
            self.M_Dirichlet = self.num_of_vertices
        
        self.num_of_panels_Dirichlet = self.num_of_panels
        self.num_of_panels_Neumann = 0

        self.panel_types.fill(int(CellFluxType.DIRICHLET_TOBESOLVED))
        self.vertice_types.fill(int(VertAttachType.DIRICHLET_TOBESOLVED))

        np_Dirichlet_index = np.array([i for i in range(self.num_of_panels)], dtype=np.int32)
        self.Dirichlet_index = ti.field(dtype=ti.i32, shape=(self.num_of_panels,))
        self.Dirichlet_index.from_numpy(np_Dirichlet_index)

        self.Neumann_index = ti.field(dtype=ti.i32, shape=())

        # Use only degree = 1 & Has Neumann to be solved
        self.map_global_Neumann_to_local = ti.field(dtype=ti.i32, shape=())
        # Use only degree = 1 & Has Dirichlet to be solved
        np_map_global_Dirichlet_to_local = np.array([i for i in range(self.num_of_vertices)], dtype=np.int32)
        self.map_global_Dirichlet_to_local = ti.field(dtype=ti.i32, shape=(self.num_of_vertices,))
        self.map_global_Dirichlet_to_local.from_numpy(np_map_global_Dirichlet_to_local)

    def set_mixed_bvp(self):
        """
        As default, Divide the whole region as:
        Dirichlet boundary if y > 0
        Neumann boundary if y <= 0
        """
        assert(self.num_of_panels == self.panel_types.shape[0])
        assert(self.num_of_vertices == self.vertice_types.shape[0])

        self.N_Neumann = 0
        self.M_Dirichlet = 0

        np_panel_types = np.array(
            [0 for i in range(self.num_of_panels)], dtype=np.int32
        )
        np_vertice_types = np.array(
            [-1 for i in range(self.num_of_vertices)], dtype=np.int32
        )
        np_map_global_Neumann_to_local = np.array(
            [-1 for i in range(self.num_of_vertices)], dtype=np.int32
        )
        np_map_global_Dirichlet_to_local = np.array(
            [-1 for i in range(self.num_of_vertices)], dtype=np.int32
        )

        num_of_Dirichlet_panels = 0
        num_of_Neumann_panels = 0
        for i in range(self.num_of_panels):
            x = 0.0
            for j in range(self._dim):
                idx = self.panels[self._dim * i + j]
                xj = self.vertices[idx].y
                x += xj
            
            x /= self._dim
            if x > 0:
                num_of_Neumann_panels += 1
                np_panel_types[i] = int(CellFluxType.NEUMANN_TOBESOLVED)
            else:
                num_of_Dirichlet_panels += 1
                np_panel_types[i] = int(CellFluxType.DIRICHLET_TOBESOLVED)
        
        num_of_Neumann_vertices = 0
        num_of_Dirichlet_vertices = 0
        for i in range(self.num_of_panels):
            if np_panel_types[i] == int(CellFluxType.NEUMANN_TOBESOLVED):
                for j in range(self._dim):
                    vert_idx = self.panels[self._dim * i + j]
                    if np_vertice_types[vert_idx] == -1:
                        np_vertice_types[vert_idx] = int(VertAttachType.NEUMANN_TOBESOLVED)
                        np_map_global_Neumann_to_local[vert_idx] = num_of_Neumann_vertices
                        num_of_Neumann_vertices += 1
    
        for i in range(self.num_of_panels):
            if np_panel_types[i] == int(CellFluxType.DIRICHLET_TOBESOLVED):
                for j in range(self._dim):
                    vert_idx = self.panels[self._dim * i + j]
                    if np_vertice_types[vert_idx] == -1:
                        np_vertice_types[vert_idx] = int(VertAttachType.DIRICHLET_TOBESOLVED)
                        np_map_global_Dirichlet_to_local[vert_idx] = num_of_Dirichlet_vertices
                        num_of_Dirichlet_vertices += 1
        
        assert(num_of_Dirichlet_vertices + num_of_Neumann_vertices == self.num_of_vertices)
        
        # Do further checks: If a A-type panel has been surrounded by a B-type panel, you have to see this panel as B-type
        for i in range(self.num_of_panels):
            Dirichlet_sum = 0
            Neumann_sum = 0
            panel_type = np_panel_types[i]
            for j in range(self._dim):
                vert_idx = self.panels[self._dim * i + j]
                if np_vertice_types[vert_idx] == int(VertAttachType.DIRICHLET_TOBESOLVED):
                    Dirichlet_sum += 1
                elif np_vertice_types[vert_idx] == int(VertAttachType.NEUMANN_TOBESOLVED):
                    Neumann_sum += 1
            
            if Neumann_sum == self._dim and panel_type == int(CellFluxType.DIRICHLET_TOBESOLVED):
                np_panel_types[i] = int(CellFluxType.NEUMANN_TOBESOLVED)
                num_of_Dirichlet_panels -= 1
                num_of_Neumann_panels += 1
            elif Dirichlet_sum == self._dim and panel_type == int(CellFluxType.NEUMANN_TOBESOLVED):
                np_panel_types[i] = int(CellFluxType.DIRICHLET_TOBESOLVED)
                num_of_Dirichlet_panels += 1
                num_of_Neumann_panels -= 1
            elif Neumann_sum != self._dim and Dirichlet_sum != self._dim:
                if panel_type == int(CellFluxType.NEUMANN_TOBESOLVED):
                    num_of_Neumann_panels -= 1
                elif panel_type == int(CellFluxType.DIRICHLET_TOBESOLVED):
                    num_of_Dirichlet_panels -= 1
                np_panel_types[i] = int(CellFluxType.MIX_NONRESOLVED)
        
        if self.shape_func_degree_Neumann == 0:
            self.N_Neumann = num_of_Neumann_panels
        
        if self.shape_func_degree_Dirichlet == 0:
            self.M_Dirichlet = num_of_Dirichlet_panels
        
        if self.shape_func_degree_Neumann == 1:
            self.N_Neumann = num_of_Neumann_vertices
        
        if self.shape_func_degree_Dirichlet == 1:
            self.M_Dirichlet = num_of_Dirichlet_vertices
        
        assert(num_of_Dirichlet_vertices + num_of_Neumann_vertices == self.num_of_vertices)

        self.num_of_panels_Dirichlet = num_of_Dirichlet_panels
        self.num_of_panels_Neumann = num_of_Neumann_panels
        
        self.panel_types.from_numpy(np_panel_types)
        self.vertice_types.from_numpy(np_vertice_types)

        np_hlp_indices = np.array([i for i in range(self.num_of_panels)], dtype=np.int32)
        np_Dirichlet_index = np_hlp_indices[np_panel_types != int(CellFluxType.NEUMANN_TOBESOLVED)]
        np_Neumann_index = np_hlp_indices[np_panel_types != int(CellFluxType.DIRICHLET_TOBESOLVED)]

        self.Dirichlet_index = ti.field(dtype=ti.i32, shape=(self.num_of_panels - num_of_Neumann_panels,))
        self.Neumann_index = ti.field(dtype=ti.i32, shape=(self.num_of_panels - num_of_Dirichlet_panels,))
        self.Dirichlet_index.from_numpy(np_Dirichlet_index)
        self.Neumann_index.from_numpy(np_Neumann_index)

        self.map_global_Neumann_to_local = ti.field(dtype=ti.i32, shape=(self.num_of_vertices,))
        self.map_global_Neumann_to_local.from_numpy(np_map_global_Neumann_to_local)
        self.map_global_Dirichlet_to_local = ti.field(dtype=ti.i32, shape=(self.num_of_vertices,))
        self.map_global_Dirichlet_to_local.from_numpy(np_map_global_Dirichlet_to_local)

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
    def map_local_Dirichlet_index_to_panel_index(self, i):
        result = -1
        if ti.static(self.M_Dirichlet > 0):
            result = self.Dirichlet_index[i]
        return result
    
    @ti.func
    def map_local_Neumann_index_to_panel_index(self, i):
        result = -1
        if ti.static(self.N_Neumann > 0):
            result = self.Neumann_index[i]
        return result
    
    @ti.func
    def map_global_vert_index_to_local_Dirichlet(self, i):
        result = -1
        if ti.static(self.M_Dirichlet > 0 and self.shape_func_degree_Dirichlet == 1):
            if self.vertice_types[i] == int(VertAttachType.DIRICHLET_TOBESOLVED) or \
                self.vertice_types[i] == int(VertAttachType.BOTH_TOBESOLVED):
                result = self.map_global_Dirichlet_to_local[i]
        return result
    
    @ti.func
    def map_global_vert_index_to_local_Neumann(self, i):
        result = -1
        if ti.static(self.N_Neumann > 0 and self.shape_func_degree_Neumann == 1):
            if self.vertice_types[i] == int(VertAttachType.NEUMANN_TOBESOLVED) or \
                self.vertice_types[i] == int(VertAttachType.BOTH_TOBESOLVED):
                result = self.map_global_Neumann_to_local[i]
        return result
    
    @ti.func
    def get_vertices(self):
        return self.vertices

    @ti.func
    def get_panels(self):
        return self.panels

    @ti.func
    def get_panel_areas(self):
        return self.panel_areas
    
    @ti.func
    def get_vert_areas(self):
        return self.vert_areas
    
    @ti.func
    def get_vert_normals(self):
        return self.vert_normals
    
    @ti.func
    def get_panel_normals(self):
        return self.panel_normals
    
    @ti.func
    def get_panel_types(self):
        return self.panel_types
    
    @ti.func
    def get_vertice_types(self):
        return self.vertice_types
    
    def load_asset(
        self,
        scale = None,
        translate = None
    ):
        scene = pywavefront.Wavefront(self._object_path, collect_faces=True)
        vertices = np.array(scene.vertices, dtype=self._np_dtype)
        faces = np.array(scene.meshes[None].faces, dtype=np.int64)
        if scale is not None:
            vertices = vertices * scale
        if translate is None:
            translate = -vertices.mean(axis=0, keepdims=True)
        vertices = vertices + translate
        if scale is None:
            vertices = vertices * (1.0 / np.abs(vertices).max())
        return vertices, faces
    
    def load_analytical_sphere(self, r: float=1.0, rows: int=17, cols: int=17):
        vertices = [[0.0, -r, 0.0]]
        panels = []
        for i in range(1, rows):
            theta = math.pi * i / rows
            for j in range(cols):
                phi = 2.0 * math.pi * j / cols
                vertices.append(
                    [r * math.cos(phi) * math.sin(theta),
                     -r * math.cos(theta),
                     r * math.sin(phi) * math.sin(theta),]
                )
                if i == 1:
                    panels.append([0, 1 + (j + 1) % cols, 1 + j])
                else:
                    panels.append([1 + (i - 2) * cols + j, 1 + (i - 2) * cols + (j + 1) % cols, 1 + (i - 1) * cols + j])
                    panels.append([1 + (i - 2) * cols + (j + 1) % cols, 1 + (i - 1) * cols + (j + 1) % cols, 1 + (i - 1) * cols + j])
        
        vertices.append([0.0, r, 0.0])
        for j in range(cols):
            panels.append([1 + (rows - 2) * cols + j, 1 + (rows - 2) * cols + (j + 1) % cols, 1 + (rows - 1) * cols])
        
        return np.array(vertices, dtype=self._np_dtype), np.array(panels, dtype=np.int64)

    def load_analytical_disk(self, r: float=1.0, N: int=256):
        vertices = np.array(
            [[r * math.cos(2.0 * math.pi * i / N) for i in range(N)], 
             [r * math.sin(2.0 * math.pi * i / N) for i in range(N)]],
            dtype=self._np_dtype
        )
        
        panels = np.array(
            [[i for i in range(N)],
             [(i + 1) % N for i in range(N)]],
            dtype=np.int64
        )

        vertices = np.transpose(vertices, (1, 0))
        panels = np.transpose(panels, (1, 0))

        return vertices, panels
    
    @ti.func
    def get_panels_relation(self, i, j) -> int:
        panel_relation = int(PanelsRelation.SEPARATE)
        if i == j:
            panel_relation = int(PanelsRelation.COINCIDE)
        else:
            if self._dim == 2:
                for ii in range(self._dim):
                    edge1_vert_idx = self.panels[self._dim * i + ii]
                    for jj in range(self._dim):
                        edge2_vert_idx = self.panels[self._dim * j + jj]

                        if edge1_vert_idx == edge2_vert_idx:
                            panel_relation = int(PanelsRelation.COMMON_VERTEX)
            elif self._dim == 3:
                has_common_vertex = False
                for ii in range(self._dim):
                    edge1_vert_idx1 = self.panels[self._dim * i + ii]
                    edge1_vert_idx2 = self.panels[self._dim * i + (ii + 1) % self._dim]
                    for jj in range(self._dim):
                        edge2_vert_idx1 = self.panels[self._dim * j + jj]
                        edge2_vert_idx2 = self.panels[self._dim * j + (jj + 1) % self._dim]

                        if edge1_vert_idx1 == edge2_vert_idx1 and edge1_vert_idx2 == edge2_vert_idx2:
                            panel_relation = int(PanelsRelation.COMMON_EDGE)
                        
                        if edge1_vert_idx1 == edge2_vert_idx2 and edge1_vert_idx2 == edge2_vert_idx1:
                            panel_relation = int(PanelsRelation.COMMON_EDGE)
                        
                        if edge1_vert_idx1 == edge2_vert_idx1:
                            has_common_vertex = True
                
                if has_common_vertex and not (panel_relation == int(PanelsRelation.COMMON_EDGE)):
                    panel_relation = int(PanelsRelation.COMMON_VERTEX)
        
        return panel_relation
    
    @ti.kernel
    def run_step(self):
        pass

    def kill(self):
        self.initialized = False
        self.vertices = None
        self.panels = None
        self.panel_areas = None
        self.vert_areas = None
        self.panel_normals = None
        self.vert_normals = None
        self.panel_types = None
        self.vertice_types = None
        self.Dirichlet_index = None
        self.Neumann_index = None
        self.map_global_Neumann_to_local = None

        self._log_manager.ErrorLog("Kill the Mesh Manager")