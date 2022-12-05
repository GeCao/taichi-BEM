"""
This is a demo for validation of BEM solver.
A Helmoholtz Transmission equation is solved, with Dirichelt boundary applies.
The boundary value is given with analytical solution.
Please refer this specific problem from book: 

    Sergej Rjasanow and Olaf Steinbach. 2007. The fast solution of boundary integral equations. Springer Science & Business Media
    equation 4.32 on Page 169

"""
import sys, os
import math
import taichi as ti
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        'vis': args.vis,
    }

    demo_path = os.path.abspath(os.curdir)
    save_path = os.path.join(demo_path, "data", "plot.png")
    print(save_path)

    # BVP problem for Dirichlet Problem
    # Laplacian(u) = 0,
    # With u(x) = (a1 + b1x1)(a2 + b2x2)exp(ıκx3) as the analytical result.
    @ti.func
    def analytical_function_Dirichlet(x, sqrt_n: float = 1):
        a1 = 0.5
        b1 = 0.5
        a2 = 0.5
        b2 = 0.5
        k = args.k
        # Compute Inner trace
        expd_vec = ti.math.cexp(ti.Vector([0.0, x[2] * k * sqrt_n]))
        vec_result = (a1 + b1 * x[0]) * (a2 + b2 * x[1]) * expd_vec

        return vec_result
    
    @ti.func
    def analytical_function_Neumann(x, normal_x, sqrt_n: float = 1):
        a1 = 0.5
        b1 = 0.5
        a2 = 0.5
        b2 = 0.5
        k = args.k

        expd_vec = ti.math.cexp(ti.Vector([0.0, x[2] * k * sqrt_n]))
        du_dx1 = b1 * (a2 + b2 * x[1]) * expd_vec
        du_dx2 = (a1 + b1 * x[0]) * b2 * expd_vec
        du_dx3 = (a1 + b1 * x[0]) * (a2 + b2 * x[1]) * ti.Vector([-expd_vec.y, expd_vec.x]) * k * sqrt_n
        vec_result = du_dx1 * normal_x.x + du_dx2 * normal_x.y + du_dx3 * normal_x.z
        return vec_result
    
    N = 10
    wave_numbers = [i * 1.0 / N for i in range(15 * N)]
    A1_inv_norms = [0.0 for i in range(len(wave_numbers))]
    A2_inv_norms = [0.0 for i in range(len(wave_numbers))]
    core_manager = CoreManager(simulation_parameters)
    core_manager.initialization(
        analytical_function_Dirichlet=analytical_function_Dirichlet,
        analytical_function_Neumann=analytical_function_Neumann
    )
    for epoch in tqdm(range(len(wave_numbers))):
        k = wave_numbers[epoch]

        core_manager._BEM_manager.matrix_layer_init()
        A1_norm = core_manager._BEM_manager.get_mat_A1_norm(k)
        A2_norm = core_manager._BEM_manager.get_mat_A2_norm(k)

        A1_inv_norms[epoch] = 1.0 * A1_norm
        A2_inv_norms[epoch] = 1.0 * A2_norm
        print("epoch = {}, A1_norm = {}, A2_norm = {}".format(epoch, A1_norm, A2_norm))
    
    fig = plt.figure(1)
    plt.title("unit sphere transmission problem")
    plt.xlabel("k")
    plt.ylabel("norm")
    plt.yscale("log")
    plt.plot(wave_numbers, A1_inv_norms, 'blue', label="A1_inv_norm")
    plt.plot(wave_numbers, A2_inv_norms, 'red', label="A2_inv_norm")
    plt.legend()
    fig.savefig(save_path)

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
        default=3,
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
        default=3,
        help="wavenumber",
    )

    parser.add_argument(
        "--boundary",
        type=str,
        default="Full",
        choices=["Dirichlet", "Neumann", "Mix", "Full"],
        help="Do we need a video for visualization?",
    )

    parser.add_argument(
        "--kernel",
        type=str,
        default="Helmholtz_Transmission",
        choices=["Laplace", "Helmholtz", "Helmholtz_Transmission"],
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

    parser.add_argument(
        "--vis",
        type=bool,
        default=False,
        help="Visualization, use GUI",
    )

    args = parser.parse_args()

    main(args)
