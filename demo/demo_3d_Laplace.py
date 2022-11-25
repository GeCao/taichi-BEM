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
        'log_to_disk': args.log_to_disk,
        'make_video': args.make_video,
        'show_wireframe': args.show_wireframe,
    }

    core_manager = CoreManager(simulation_parameters)

    # BVP problem for Dirichlet Problem
    # Laplacian(u) = 0,
    # With u(x) = (1 + x1)exp(2π x2)cos(2π x3) as the analytical result.
    @ti.func
    def analytical_function_Dirichlet(x):
        # return ti.Vector([1 + x[1]])
        return ti.Vector([(1 + x[0]) * ti.math.exp(2.0 * ti.math.pi * x[1]) * ti.math.cos(2.0 * ti.math.pi * x[2])])
    
    @ti.func
    def analytical_function_Neumann(x, normal_x):
        grad_u = ti.Vector(
            [ti.math.exp(2.0 * ti.math.pi * x[1]) * ti.math.cos(2.0 * ti.math.pi * x[2]),
             2.0 * ti.math.pi * (1 + x[0]) * ti.math.exp(2.0 * ti.math.pi * x[1]) * ti.math.cos(2.0 * ti.math.pi * x[2]),
             -2.0 * ti.math.pi * (1 + x[0]) * ti.math.exp(2.0 * ti.math.pi * x[1]) * ti.math.sin(2.0 * ti.math.pi * x[2])]
        )
        # grad_u = ti.Vector([0.0, 1.0, 0.0])
        return ti.Vector([grad_u.dot(normal_x)])
    
    core_manager.initialization(
        analyical_function_Dirichlet=analytical_function_Dirichlet,
        analyical_function_Neumann=analytical_function_Neumann
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
        choices=["sphere", "cube", "hemisphere", "stanford_bunny"],
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
        choices=["Laplace", "Helmholtz"],
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

    args = parser.parse_args()

    main(args)