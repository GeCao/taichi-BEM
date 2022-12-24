"""
This is a demo for validation of BEM solver.
A Laplace equation is solved, with Dirichelt boundary applies.
The boundary value is given with analytical solution.
Please refer this specific problem from book: 

    Sergej Rjasanow and Olaf Steinbach. 2007. The fast solution of boundary integral equations. Springer Science & Business Media
    equation 4.13 on Page 143

"""
import sys, os
import math
import taichi as ti
import argparse

sys.path.append("../")

from src.managers import CoreManager


ti.init(arch=ti.gpu, kernel_profiler=True)

def main(args):
    simulation_parameters = {
        'dim': args.dim,
        'object': args.object,
        'GaussQR': args.GaussQR,
        'n_i': args.n_i,
        'n_o': args.n_o,
        'k': args.k,
        'kernel': args.kernel,
        'boundary': args.boundary,
        'scope': args.scope,
        'show': args.show,
        'Q_Neumann': args.Q_Neumann,
        'Q_Dirichlet': args.Q_Dirichlet,
        'log_to_disk': args.log_to_disk,
        'make_video': args.make_video,
        'show_wireframe': args.show_wireframe,
        'vis': args.vis,
    }

    core_manager = CoreManager(simulation_parameters)

    # BVP problem for Dirichlet Problem
    # Laplacian(u) = 0,
    # With u(x) = (1 + x1)exp(2π x2)cos(2π x3) as the analytical result.
    @ti.func
    def analytical_function_Dirichlet_3d(x):
        return ti.Vector([1 + x.y - x.x - x.z])
        # return ti.Vector([(1 + x[0]) * ti.math.exp(2.0 * ti.math.pi * x[1]) * ti.math.cos(2.0 * ti.math.pi * x[2])])
    
    @ti.func
    def analytical_function_Neumann_3d(x, normal_x):
        # grad_u = ti.Vector(
        #     [ti.math.exp(2.0 * ti.math.pi * x[1]) * ti.math.cos(2.0 * ti.math.pi * x[2]),
        #      2.0 * ti.math.pi * (1 + x[0]) * ti.math.exp(2.0 * ti.math.pi * x[1]) * ti.math.cos(2.0 * ti.math.pi * x[2]),
        #      -2.0 * ti.math.pi * (1 + x[0]) * ti.math.exp(2.0 * ti.math.pi * x[1]) * ti.math.sin(2.0 * ti.math.pi * x[2])]
        # )
        grad_u = ti.Vector([-1.0, 1.0, -1.0])
        return ti.Vector([grad_u.dot(normal_x)])
    

    @ti.func
    def analytical_function_Dirichlet_2d(x):
        return ti.Vector([0.0 + 0.5 * x.x * x.x - 0.5 * x.y * x.y])
    
    @ti.func
    def analytical_function_Neumann_2d(x, normal_x):
        grad_u = ti.Vector([x.x, -x.y])
        return ti.Vector([grad_u.dot(normal_x)])
    
    if simulation_parameters["dim"] == 2:
        core_manager.initialization(
            analytical_function_Dirichlet=analytical_function_Dirichlet_2d,
            analytical_function_Neumann=analytical_function_Neumann_2d
        )
    elif simulation_parameters["dim"] == 3:
        core_manager.initialization(
            analytical_function_Dirichlet=analytical_function_Dirichlet_3d,
            analytical_function_Neumann=analytical_function_Neumann_3d
        )
    core_manager.run()
    core_manager.kill()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=3,
        choices=[2, 3],
        help="dimension: 2D or 3D",
    )

    parser.add_argument(
        "--object",
        type=str,
        default="sphere",
        choices=["sphere", "cube", "hemisphere", "stanford_bunny", "disk", "suzan"],
        help="dimension: 2D or 3D",
    )

    parser.add_argument(
        "--GaussQR",
        type=int,
        default=7,
        help="Gauss QR number",
    )

    parser.add_argument(
        "--n_i",
        type=int,
        default=1,
        help="n_i, physical if n_i < n_o",
    )

    parser.add_argument(
        "--n_o",
        type=int,
        default=1,
        help="n_o, physical if n_i < n_o",
    )

    parser.add_argument(
        "--k",
        type=float,
        default=0,
        help="wavenumber",
    )

    parser.add_argument(
        "--kernel",
        type=str,
        default="Laplace",
        choices=["Laplace", "Helmholtz", "Helmholtz_Transmission"],
        help="Do we need a video for visualization?",
    )

    parser.add_argument(
        "--boundary",
        type=str,
        default="Dirichlet",
        choices=["Dirichlet", "Neumann", "Mix"],
        help="Do we need a video for visualization?",
    )

    parser.add_argument(
        "--scope",
        type=str,
        default="Interior",
        choices=["Interior", "Exterior"],
        help="Indicating Interior or Exterior, not useful for transmission problem",
    )

    parser.add_argument(
        "--show",
        type=str,
        default="Default",
        choices=["Default", "Neumann", "Dirichlet"],
        help="Usually we apply Neumann(Dirichlet) boundary and solve Dirichlet(Neumann), "
             "we just need to show what we solved in this (Default) case."
             "However, sometimes we solve both, in this case, you need to indicate one for visualization",
    )

    parser.add_argument(
        "--Q_Neumann",
        type=int,
        default=0,
        help="The degree of Neumann attached shape function",
    )

    parser.add_argument(
        "--Q_Dirichlet",
        type=int,
        default=1,
        help="The degree of Dirichlet attached shape function",
    )

    parser.add_argument(
        "--log_to_disk",
        type=bool,
        default=False,
        help="Do we need a Log for every run?",
    )

    parser.add_argument(
        "--make_video",
        type=bool,
        default=False,
        help="Do we need a video for visualization?",
    )

    parser.add_argument(
        "--show_wireframe",
        type=bool,
        default=False,
        help="Do we need a video for visualization?",
    )

    parser.add_argument(
        "--vis",
        type=bool,
        default=True,
        help="Visualization, use GUI",
    )

    args = parser.parse_args()

    main(args)