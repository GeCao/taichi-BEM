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
        'n_i': args.n_i,
        'n_o': args.n_o,
        'k': args.k,
        'kernel': args.kernel,
        'log_to_disk': args.log_to_disk,
        'make_video': args.make_video,
        'show_wireframe': args.show_wireframe,
    }

    core_manager = CoreManager(simulation_parameters)

    # BVP problem for Dirichlet Problem
    # Laplacian(u) = 0,
    # With u(x) = (a1 + b1x1)(a2 + b2x2)exp(ıκx3) as the analytical result.
    @ti.func
    def analytical_function_Dirichlet(k, x):
        a1 = 0.5
        b1 = 0.5
        a2 = 0.5
        b2 = 0.5
        expd_vec = ti.math.cexp(ti.Vector[0.0, x[2] * k])
        expd_number = expd_vec.x + expd_vec.y * 1j
        vec_result = (a1 + b1 * x[0]) * (a2 * b2 * x[1]) * expd_vec
        return vec_result
    
    @ti.func
    def analytical_function_Neumann(k, x, normal_x):
        a1 = 0.5
        b1 = 0.5
        a2 = 0.5
        b2 = 0.5
        expd_vec = ti.math.cexp(ti.Vector[0.0, x[2] * k])
        expd_number = expd_vec.x + expd_vec.y * 1j
        grad_u = [
            b1 * (a2 * b2 * x[1]) * expd_number,
            (a1 * b1 * x[1]) * b2 * expd_number,
            (a1 + b1 * x[0]) * (a2 * b2 * x[1]) * expd_number * k * 1j
        ]
        complex_result = grad_u[0] * normal_x.x + grad_u[1] * normal_x.y + grad_u[2] * normal_x.z
        vec_result = ti.Vector([complex_result.x, complex_result.y])
        return vec_result
    
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
        default=1,
        help="wavenumber",
    )

    parser.add_argument(
        "--kernel",
        type=str,
        default="Helmholtz",
        choices=["Laplace", "Helmholtz"],
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
        default=True,
        help="Do we need a video for visualization?",
    )

    args = parser.parse_args()

    main(args)
