# Taichi-BEM

### ----------Analytical(Left)----Solved(Middle)----Difference(Right)

### Laplace Equation
<img src="demo/Laplace_solved_Neumann.png" height="270">

### Helmholtz Equation
<img src="demo/Helmholtz_solved_Dirichlet.png" height="270">

### Helmholtz Transmission Equation (Still Working on! 👨‍💻)
<img src="demo/HelmholtzTransmission_solved_Full.png" height="270">


### How to run the code

```bash
cd taichi-BEM
pip install -r requirements.txt
cd demo
python demo_3d_Laplace.py --boundary Dirichlet
python demo_3d_Helmholtz.py --boundary Neumann
python demo_3d_Helmholtz_Transmission.py --boundary Neumann --k 3.14 
```

### Parameters
- boundary := [‘Dirichlet’, 'Neumann', 'Mix']
  - Dirichlet: If You want to set Dirichlet boundary and solve Neumann charges, pick this.
  - Neumann: If You want to set Neumann boundary and solve Dirichlet charges, pick this.
  - Mix: If You want to set Dirichlet boundary and solve Neumann charges for y > 0, and otherwise for y <= 0, pick this. If you want to self define your design of Dirichlet region and Neumann region, go to file [mesh_manager.py](src/managers/mesh_manager.py) and reach function [set_mixed_bvp](src/managers/mesh_manager.py), you might handle it with your own preference.
- kernel := ['Laplace', 'Helmholtz']
  - Laplace: Solve Laplace equation
  - Helmholtz: Solve Helmholtz equation
- object := ['sphere', 'cube']
  - sphere: an obj file will be read into our scope by [sphere.obj](assets/sphere.obj)
  - You can add your own .obj files into [/assets](assets/) directory and use the obj file name as your own mesh object
- show_wireframe := [True, False]
  - This parameter defines if wireframe is shown in final GUI


### Trouble Shooting