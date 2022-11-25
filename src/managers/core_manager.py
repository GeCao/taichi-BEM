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
        if self._simulation_parameters['kernel'] != "Helmholtz":
            self._simulation_parameters['k'] = 0
        self._np_dtype = np.float32
        self._ti_dtype = ti.f32
        self.log_to_disk = self._simulation_parameters["log_to_disk"]
        self.make_video = self._simulation_parameters["make_video"]
        self.show_wireframe = self._simulation_parameters["show_wireframe"]
        self.res = (1920, 640)

        self._log_manager = LogManager(self, self.log_to_disk)
        self._mesh_manager = MeshManager(self)
        self._BEM_manager = BEMManager(self)

        self._video_manager = ti.tools.VideoManager(output_dir=self.data_path, framerate=24, automatic_build=False)

        self.initialized = False
    
    def initialization(self, analyical_function_Dirichlet, analyical_function_Neumann):
        self._log_manager.initialization()
        self._mesh_manager.initialization()
        self._BEM_manager.initialization(analyical_function_Dirichlet, analyical_function_Neumann)

        self.window = ti.ui.Window('BEM Example: Left=Analytical, Right=Solved', res=self.res, pos=(150, 150), vsync=True)
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((1.0, 1.0, 1.0))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(5.4, 0.0, 0.0)
        self.camera.lookat(0.0, 0.0, 0.0)
        self.camera.up(0.0, 1.0, 0.0)

        self.iteration = 0

        self.initialized = True

    def run(self):
        self._BEM_manager.run()
        self._log_manager.InfoLog("BEM has finished running")

        while self.window.running:
            self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.LMB)
            self.scene.set_camera(self.camera)
            self.scene.ambient_light((0.8, 0.8, 0.8))
            self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

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
        
        if self.make_video:
            self._video_manager.make_video(gif=False, mp4=True)

    def kill(self):
        self.initialized = False
        self.iteration = 0
        self._BEM_manager.kill()
        self._mesh_manager.kill()
        self._log_manager.kill()
