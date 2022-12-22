import os, sys
import taichi as ti
import numpy as np
import math
import cv2

from src.managers import LogManager, MeshManager, BEMManager


@ti.data_oriented
class CoreManager:
    def __init__(self, simulation_parameters):
        self.demo_path = os.path.abspath(os.curdir)
        self.root_path = os.path.abspath(os.path.join(self.demo_path, '..'))
        self.data_path = os.path.join(self.demo_path, 'data')
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        print("The root path of our project: ", self.root_path)
        print("The data path of our project: ", self.data_path)

        self._simulation_parameters = simulation_parameters
        if "Helmholtz" not in self._simulation_parameters['kernel']:
            self._simulation_parameters['k'] = 0
        
        self._is_transmission = 1 if "Full" in self._simulation_parameters['boundary'] or "Transmission" in self._simulation_parameters['kernel'] else 0
        self._np_dtype = np.float32
        self._ti_dtype = ti.f32
        if self._simulation_parameters["dim"] == 2:
            self._np_dtype = np.float64
            self._ti_dtype = ti.f64
        
        self.log_to_disk = self._simulation_parameters["log_to_disk"]
        self.make_video = self._simulation_parameters["make_video"]
        self.show_wireframe = self._simulation_parameters["show_wireframe"]
        self.vis = self._simulation_parameters["vis"]
        self.res = (1280, 640)

        self._log_manager = LogManager(self, self.log_to_disk)
        self._mesh_manager = MeshManager(self)
        self._BEM_manager = BEMManager(self)

        self._video_manager = ti.tools.VideoManager(output_dir=self.data_path, framerate=24, automatic_build=False)

        self.initialized = False
    
    def initialization(self, analytical_function_Dirichlet, analytical_function_Neumann):
        self._log_manager.initialization()
        self._mesh_manager.initialization()
        self._BEM_manager.initialization(analytical_function_Dirichlet, analytical_function_Neumann)

        self.iteration = 0

        self.initialized = True

    def run(self):
        self._BEM_manager.run()
        self._log_manager.InfoLog("BEM has finished running")

        if self.vis:
            self.window = ti.ui.Window('BEM Example: Left=Analytical, Right=Solved', res=self.res, pos=(150, 150), vsync=True)
            self.canvas = self.window.get_canvas()
            self.canvas.set_background_color((1.0, 1.0, 1.0))
            self.scene = ti.ui.Scene()
            self.camera = ti.ui.Camera()
            self.camera.position(0.0, 0.0, 5.4)
            self.camera.lookat(0.0, 0.0, 0.0)
            self.camera.up(0.0, 1.0, 0.0)
            
            while self.window.running:
                if self.iteration <= 0:
                    torch_solved = self._BEM_manager.solved_vert_color.to_torch().norm(dim=-1)
                    torch_analytical_solved = self._BEM_manager.analytical_vert_color.to_torch().norm(dim=-1)
                    torch_diff_solved = self._BEM_manager.diff_vert_color.to_torch().norm(dim=-1)
                    self._log_manager.InfoLog("analytical sovle min = {}, max = {}, mean = {}".format(
                        torch_analytical_solved.min(), torch_analytical_solved.max(), torch_analytical_solved.mean())
                    )
                    self._log_manager.InfoLog("solve min = {}, max = {}, mean = {}".format(
                        torch_solved.min(), torch_solved.max(), torch_solved.mean())
                    )
                    self._log_manager.InfoLog("residual min = {}, max = {}, mean = {}".format(
                        torch_diff_solved.min(), torch_diff_solved.max(), torch_diff_solved.mean())
                    )
                self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.LMB)
                self.scene.set_camera(self.camera)
                self.scene.ambient_light((0.8, 0.8, 0.8))
                self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

                if self._BEM_manager._dim == 2:
                    self.scene.particles(
                        self._BEM_manager.solved_vertices,
                        radius=0.03,
                        per_vertex_color=self._BEM_manager.solved_vert_color,
                    )

                    self.scene.particles(
                        self._BEM_manager.analytical_vertices,
                        radius=0.03,
                        per_vertex_color=self._BEM_manager.analytical_vert_color,
                    )

                    self.scene.particles(
                        self._BEM_manager.diff_vertices,
                        radius=0.03,
                        per_vertex_color=self._BEM_manager.diff_vert_color,
                    )
                else:
                    self.scene.mesh(
                        self._BEM_manager.solved_vertices,
                        self._mesh_manager.panels,
                        per_vertex_color=self._BEM_manager.solved_vert_color,
                        two_sided=True,
                        show_wireframe=self.show_wireframe,
                    )

                    self.scene.mesh(
                        self._BEM_manager.analytical_vertices,
                        self._mesh_manager.panels,
                        per_vertex_color=self._BEM_manager.analytical_vert_color,
                        two_sided=True,
                        show_wireframe=self.show_wireframe,
                    )

                    self.scene.mesh(
                        self._BEM_manager.diff_vertices,
                        self._mesh_manager.panels,
                        per_vertex_color=self._BEM_manager.diff_vert_color,
                        two_sided=True,
                        show_wireframe=self.show_wireframe,
                    )

                self.canvas.scene(self.scene)

                if self.make_video:
                    img = self.window.get_image_buffer_as_numpy()
                    self._video_manager.write_frame(img)
                
                self.iteration += 1
                
                self.window.show()

    def kill(self):
        if self.make_video:
            self._video_manager.make_video(gif=False, mp4=True)
        
        self.initialized = False
        self.iteration = 0
        self._BEM_manager.kill()
        self._mesh_manager.kill()
        self._log_manager.kill()
