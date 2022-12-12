# Taichi-BEM

### ----------Analytical(Left)----Solved(Middle)----Difference(Right)

### Laplace Equation
<img src="demo/Laplace_solved_Neumann.png" height="270">

### Helmholtz Equation
<img src="demo/Helmholtz_solved_Dirichlet.png" height="270">

### Helmholtz Transmission Equation
<img src="demo/HelmholtzTransmission_solved_Full.png" height="270">

### Norm analysis (Still Working on! 👨‍💻)
* The sample step of wave number is 1/30, which implies for wavenumbers between [16, 18], 60 points are sampled.
* Precious matters, float64 shows more stability than float32 for operator AI (Somehow not very obvious for operator AII).
### The norm of AI (Left) and Augmented AI (Right)
* Although I forgot to put them together into one figure, it is still obviously the right figure (Augmented) averagely 10 times smaller, as well as holding shorted peaks.

<img src="demo/A1_plot_Neumann1_Dirichlet1.png" height="192"> <img src="demo/A1_plot_augment_1_1.png" height="192">

### The norm of AII (Left) and Augmented AI (Right)
* Samilar as above

<img src="demo/A2_plot_Neumann1_Dirichlet1.png" height="192"> <img src="demo/A2_plot_augment_1_1.png" height="192">

### How to run the code

```bash
cd taichi-BEM
pip install -r requirements.txt
cd demo
python demo_3d_Laplace.py --boundary Dirichlet
python demo_3d_Helmholtz.py --boundary Neumann --k 3
python demo_3d_Helmholtz_Transmission.py --boundary Full --k 5
```

### Parameters
- boundary := [‘Dirichlet’, 'Neumann', 'Mix']
  - Dirichlet: If You want to set Dirichlet boundary and solve Neumann charges, pick this.
  - Neumann: If You want to set Neumann boundary and solve Dirichlet charges, pick this.
  - Mix: If You want to set Dirichlet boundary and solve Neumann charges for y > 0, and otherwise for y <= 0, pick this. If you want to self define your design of Dirichlet region and Neumann region, go to file [mesh_manager.py](src/managers/mesh_manager.py) and reach function [set_mixed_bvp](src/managers/mesh_manager.py), you might handle it with your own preference.
  - Full: This is only useful for Transmission problem, Both Neumann/Dirichlet boundary will be applied, and the corresponding boundary will be solved for interior.
- kernel := ['Laplace', 'Helmholtz']
  - Laplace: Solve Laplace equation
  - Helmholtz: Solve Helmholtz equation
- object := ['sphere', 'cube']
  - sphere: an obj file will be read into our scope by [sphere.obj](assets/sphere.obj)
  - You can add your own .obj files into [/assets](assets/) directory and use the obj file name as your own mesh object
- show_wireframe := [True, False]
  - This parameter defines if wireframe is shown in final GUI


### Trouble Shooting