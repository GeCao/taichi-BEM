import taichi as ti

from src.BEM_functions.double_layer import AbstractDoubleLayer
from src.BEM_functions.utils import CellFluxType, VertAttachType, KernelType, PanelsRelation


@ti.data_oriented
class DoubleLayer3d(AbstractDoubleLayer):
    rank = 3

    def __init__(self, BEM_manager, *args, **kwargs,):
        super(DoubleLayer3d, self).__init__(BEM_manager, *args, **kwargs)

        self._GaussQR = self._BEM_manager.get_GaussQR()
        self._dim = self._BEM_manager._dim

        self._ti_dtype = self._BEM_manager._ti_dtype
        self._np_dtype = self._BEM_manager._np_dtype
        self._n = self._BEM_manager._n

        self.N_Neumann = self._BEM_manager.get_N_Neumann()
        self._Q_Neumann = self._BEM_manager.get_shape_func_degree_Neumann()
        self.M_Dirichlet = self._BEM_manager.get_M_Dirichlet()
        self._Q_Dirichlet = self._BEM_manager.get_shape_func_degree_Dirichlet()
        self.num_of_vertices = self._BEM_manager.get_num_of_vertices()
        self.num_of_panels = self._BEM_manager.get_num_of_panels()
        self.num_of_panels_Neumann = self._BEM_manager.get_num_of_panels_Neumann()
        self.num_of_panels_Dirichlet = self._BEM_manager.get_num_of_panels_Dirichlet()

        self._Kmat = ti.Vector.field(self._n, dtype=self._ti_dtype, shape=())
        assert(self.N_Neumann + self.M_Dirichlet > 0)
        if self.N_Neumann > 0 and self.M_Dirichlet > 0:
            self._Kmat = ti.Vector.field(
                self._n,
                dtype=self._ti_dtype,
                shape=(self.N_Neumann, self.M_Dirichlet)
            )
    
    @ti.func
    def get_K_mat(self):
        return self._Kmat
    
    def kill(self):
        self.N_Neumann = 0
        self.M_Dirichlet = 0
        self._Q_Neumann = -1
        self._Q_Dirichlet = -1
        self.num_of_vertices = 0
        self.num_of_panels = 0
        self.num_of_panels_Neumann = 0
        self.num_of_panels_Dirichlet = 0

        self._Kmat = None
    
    @ti.func
    def integrate_on_two_panels(
        self,
        rands: ti.types.vector(4, int),
        scope_type: int,
        k: float,
        sqrt_n: float,
        panel_x: int,
        panel_y: int,
        basis_function_index_x: int,
        basis_function_index_y: int,
        panels_relation: int
    ):
        """
        Get Integration points and weights for a general triangle (Use Duffy Transform)
        ### Perspective from Duffy Transform
         y
         ^
        1|__________
         |          |
         |  *    *  |
         |          |     <- This refers GaussQR = 2, you will get GaussQR*GaussQR=4 points and 4 weights
         |  *    *  |
        0-----------|1--->x
         0

         ============= TO =============

         y
         ^
        1|
         |   /|
         |  / |
         | /  |           <- Points: (x, y) := (x, x * y)
         |/   |           <- Weights:   w  To  w
        0|----|1--->x     <- Jaobian:   1   To  x
         0

         ============= TO =============
          r2
         ^
        1|                                      x2
         |   /|                                 /|
         |  / |                                / |
         | /  |                      ->       /  |
         |/   |                              /   |
        0|----|1--->r1                    x3/____|x1
         0
         
         - How to project a unit triangle (x1, x2, x3) to a general one?
         - If we choose (0, 0)->x1, (1, 0)->x2, (1, 1)->x3
         - x (r1, r2) = (1 - r1) * x1 + (r1 - r2) * x2 + r2 * x3

         ==============================
        To sum up, a random number (t, s) as a sample on triangle, where t,s \in [0, 1]
        r1, r2          = ...
        Sampled_point   = (1 - r1) * x1  +  (r1 - r2) * x2  +  r2 * x3
        Corres_weight   = w * 2 * Area
        Jacobian        = r1
        phase functions = 1 - r1, r1 - r2, r2

        !!!!!!!!!!!!!!
        However, if these two panels has overlaps, such like common edegs, common vertices, even the same panel.
        For instance, the same panel:
        Reference: S. A. Sauter and C. Schwab, Boundary Element Methods, Springer Ser. Comput. Math. 39, Springer-Verlag, Berlin, 2011.
        https://link.springer.com/content/pdf/10.1007/978-3-540-68093-2.pdf
        See this algorithm from chapter 5.2.1

        Get Integration on coincide triangles, where
        int_{Tau_x} int_{Tau_x} (func) dx dy
            = (2 * area_x) * (2 * area_y) *           int_{unit_panel}    int_{unit_panel}     (func) dx             dy
            = (2 * area_x) * (2 * area_y) * sum_{6} * int_{0->1} int_{0->w1} int_{0->w2} int_{0->w3} (func) d(w1) d(w2)    d(w3) d(w4)
            = (2 * area_x) * (2 * area_y) * sum_{6} * int_{0->1} int_{0->1}  int_{0->1}  int_{0->1}  (func) d(xsi) d(eta1) d(eta2) d(eta3)
        
        Similarly, besides the coincide case we have mentioned above,
        other approaches such like common vertices/edges can be refered by chapter 5.2.2 and 5.2.2
        """
        integrand = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)

        x1 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * panel_x + 0)
        x2 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * panel_x + 1)
        x3 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * panel_x + 2)
        area_x = self._BEM_manager.get_panel_area(panel_x)
        normal_x = self._BEM_manager.get_panel_normal(panel_x, scope_type=scope_type)

        y1 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * panel_y + 0)
        y2 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * panel_y + 1)
        y3 = self._BEM_manager.get_vertice_from_flat_panel_index(self._dim * panel_y + 2)
        area_y = self._BEM_manager.get_panel_area(panel_y)
        normal_y = self._BEM_manager.get_panel_normal(panel_y, scope_type=scope_type)

        # Generate number(xsi, eta1, eta2, eta3)
        xsi = self._BEM_manager.Gauss_points_1d[rands.x]
        eta1 = self._BEM_manager.Gauss_points_1d[rands.y]
        eta2 = self._BEM_manager.Gauss_points_1d[rands.z]
        eta3 = self._BEM_manager.Gauss_points_1d[rands.w]

        # Scale your weight
        weight_x = self._BEM_manager.Gauss_weights_1d[rands.x] * self._BEM_manager.Gauss_weights_1d[rands.y] * (area_x * 2.0)
        weight_y = self._BEM_manager.Gauss_weights_1d[rands.z] * self._BEM_manager.Gauss_weights_1d[rands.w] * (area_y * 2.0)
        
        weight = weight_x * weight_y

        if panels_relation == int(PanelsRelation.SEPARATE):
            # Generate number(r1, r2) for panel x
            r1_x = xsi
            r2_x = eta1 * r1_x
            # Generate number(r1, r2) for panel y
            r1_y = eta2
            r2_y = eta3 * r1_y

            # Get your jacobian
            jacobian = r1_x * r1_y

            x = self.interplate_from_unit_panel_to_general(
                r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
            )
            y = self.interplate_from_unit_panel_to_general(
                r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
            )
            phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
            phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

            integrand += (
                self.grad_G_y(x=x, y=y, normal_y=normal_y, k=k, sqrt_n=sqrt_n) * phix * phiy
            ) * weight * jacobian
        elif panels_relation == int(PanelsRelation.COINCIDE):
            # Get your jacobian
            jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2

            # This algorithm includes 6 regions D1 ~ D6
            # By symmetic of kernel, we can simply compress it into 3 regions
            w = ti.Vector([xsi, xsi * eta1, xsi * eta1 * eta2, xsi * eta1 * eta2 * eta3])
            for iiii in range(self._BEM_manager.m_mats_coincide.shape[0]):
                xz = self._BEM_manager.m_mats_coincide[iiii] @ w  # On unit triangle
                
                r1_x, r2_x = xz[0], xz[1]
                r1_y, r2_y = xz[0] - xz[2], xz[1] - xz[3]

                x = self.interplate_from_unit_panel_to_general(
                    r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                )
                y = self.interplate_from_unit_panel_to_general(
                    r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                )
                phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

                # D1, D3, D5
                integrand += (
                    self.grad_G_y(x=x, y=y, normal_y=normal_y, k=k, sqrt_n=sqrt_n) * phix * phiy
                ) * weight * jacobian

                # D2, D4, D6
                integrand += (
                    self.grad_G_y(x=y, y=x, normal_y=normal_y, k=k, sqrt_n=sqrt_n) * phix * phiy
                ) * weight * jacobian

                # r1_y, r2_y = xz[0], xz[1]
                # r1_x, r2_x = xz[0] - xz[2], xz[1] - xz[3]

                # x = self.interplate_from_unit_panel_to_general(
                #     r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
                # )
                # y = self.interplate_from_unit_panel_to_general(
                #     r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
                # )
                # phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
                # phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

                # # D2, D4, D6
                # integrand += (
                #     self.grad_G_y(x=x, y=y, normal_y=normal_y, k=k, sqrt_n=sqrt_n) * phix * phiy
                # ) * weight * jacobian
        elif panels_relation == int(PanelsRelation.COMMON_VERTEX):
            # This algorithm includes 6 regions D1, D2
            # D1
            w = ti.Vector(
                [xsi, xsi * eta1, xsi * eta2, xsi * eta2 * eta3]
            )
            r1_x, r2_x = w[0], w[1]
            r1_y, r2_y = w[2], w[3]
            x = self.interplate_from_unit_panel_to_general(
                r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
            )
            y = self.interplate_from_unit_panel_to_general(
                r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
            )
            phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
            phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

            jacobian = xsi * xsi * xsi * eta2
            integrand += (
                self.grad_G_y(x=x, y=y, normal_y=normal_y, k=k, sqrt_n=sqrt_n) * phix * phiy
            ) * weight * jacobian

            # D2
            w = ti.Vector(
                [xsi * eta2, xsi * eta2 * eta3, xsi, xsi * eta1]
            )
            r1_x, r2_x = w[0], w[1]
            r1_y, r2_y = w[2], w[3]
            x = self.interplate_from_unit_panel_to_general(
                r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
            )
            y = self.interplate_from_unit_panel_to_general(
                r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
            )
            phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
            phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

            jacobian = xsi * xsi * xsi * eta2
            integrand += (
                self.grad_G_y(x=x, y=y, normal_y=normal_y, k=k, sqrt_n=sqrt_n) * phix * phiy
            ) * weight * jacobian
        elif panels_relation == int(PanelsRelation.COMMON_EDGE):
            # This algorithm includes 6 regions D1 ~ D5
            # D1
            w = ti.Vector(
                [xsi, -xsi * eta1 * eta2, xsi * eta1 * (1.0 - eta2), xsi * eta1 * eta3]
            )
            r1_x, r2_x = w[0], w[3]
            r1_y, r2_y = r1_x + w[1], w[2]
            x = self.interplate_from_unit_panel_to_general(
                r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
            )
            y = self.interplate_from_unit_panel_to_general(
                r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
            )
            phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
            phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

            jacobian = xsi * xsi * xsi * eta1 * eta1
            integrand += (
                self.grad_G_y(x=x, y=y, normal_y=normal_y, k=k, sqrt_n=sqrt_n) * phix * phiy
            ) * weight * jacobian

            # D2
            w = ti.Vector(
                [xsi, -xsi * eta1 * eta2 * eta3, xsi * eta1 * eta2 * (1.0 - eta3), xsi * eta1]
            )
            r1_x, r2_x = w[0], w[3]
            r1_y, r2_y = r1_x + w[1], w[2]
            x = self.interplate_from_unit_panel_to_general(
                r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
            )
            y = self.interplate_from_unit_panel_to_general(
                r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
            )
            phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
            phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

            jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
            integrand += (
                self.grad_G_y(x=x, y=y, normal_y=normal_y, k=k, sqrt_n=sqrt_n) * phix * phiy
            ) * weight * jacobian

            # D3
            w = ti.Vector(
                [xsi * (1.0 - eta1 * eta2), xsi * eta1 * eta2, xsi * eta1 * eta2 * eta3, xsi * eta1 * (1.0 - eta2)]
            )
            r1_x, r2_x = w[0], w[3]
            r1_y, r2_y = r1_x + w[1], w[2]
            x = self.interplate_from_unit_panel_to_general(
                r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
            )
            y = self.interplate_from_unit_panel_to_general(
                r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
            )
            phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
            phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

            jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
            integrand += (
                self.grad_G_y(x=x, y=y, normal_y=normal_y, k=k, sqrt_n=sqrt_n) * phix * phiy
            ) * weight * jacobian

            # D4
            w = ti.Vector(
                [xsi * (1.0 - eta1 * eta2 * eta3), xsi * eta1 * eta2 * eta3, xsi * eta1, xsi * eta1 * eta2 * (1.0 - eta3)]
            )
            r1_x, r2_x = w[0], w[3]
            r1_y, r2_y = r1_x + w[1], w[2]

            x = self.interplate_from_unit_panel_to_general(
                r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
            )
            y = self.interplate_from_unit_panel_to_general(
                r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
            )
            phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
            phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)

            jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
            integrand += (
                self.grad_G_y(x=x, y=y, normal_y=normal_y, k=k, sqrt_n=sqrt_n) * phix * phiy
            ) * weight * jacobian

            # D5
            w = ti.Vector(
                [xsi * (1.0 - eta1 * eta2 * eta3), xsi * eta1 * eta2 * eta3, xsi * eta1 * eta2, xsi * eta1 * (1.0 - eta2 * eta3)]
            )
            r1_x, r2_x = w[0], w[3]
            r1_y, r2_y = r1_x + w[1], w[2]

            x = self.interplate_from_unit_panel_to_general(
                r1=r1_x, r2=r2_x, x1=x1, x2=x2, x3=x3
            )
            y = self.interplate_from_unit_panel_to_general(
                r1=r1_y, r2=r2_y, x1=y1, x2=y2, x3=y3
            )
            phix = self.shape_function(r1_x, r2_x, i=basis_function_index_x)
            phiy = self.shape_function(r1_y, r2_y, i=basis_function_index_y)
            
            jacobian = xsi * xsi * xsi * eta1 * eta1 * eta2
            integrand += (
                self.grad_G_y(x=x, y=y, normal_y=normal_y, k=k, sqrt_n=sqrt_n) * phix * phiy
            ) * weight * jacobian
        
        return integrand
    
    @ti.kernel
    def forward(self, scope_type: int, k: float, sqrt_n: float):
        """
        Compute BIO matix K_mat
        """
        if ti.static(self.N_Neumann > 0 and self.M_Dirichlet > 0):
            self._Kmat.fill(0)

            basis_func_num_Neumann = self._BEM_manager.get_num_of_basis_functions_from_Q(self._Q_Neumann)
            basis_func_num_Dirichlet = self._BEM_manager.get_num_of_basis_functions_from_Q(self._Q_Dirichlet)

            GaussQR2 = self._GaussQR * self._GaussQR
            GaussQR4 = GaussQR2 * GaussQR2

            for local_I in range(self.num_of_panels_Neumann):
                for local_J in range(self.num_of_panels_Dirichlet):
                    global_i = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_I)
                    global_j = self._BEM_manager.map_local_Dirichlet_index_to_panel_index(local_J)

                    panel_type_i = self._BEM_manager.get_panel_type(global_i)
                    panel_type_j = self._BEM_manager.get_panel_type(global_j)
                    if panel_type_i == int(CellFluxType.BOTH_TOBESOLVED):
                        panel_type_i = int(CellFluxType.NEUMANN_TOBESOLVED)
                    
                    if panel_type_j == int(CellFluxType.BOTH_TOBESOLVED):
                        panel_type_j = int(CellFluxType.DIRICHLET_TOBESOLVED)

                    panels_relation = self._BEM_manager.get_panels_relation(global_i, global_j)
                        
                    for ii in range(basis_func_num_Neumann):
                        for jj in range(basis_func_num_Dirichlet):
                            basis_function_index_x = self._BEM_manager.get_basis_function_index(self._Q_Neumann, ii)
                            basis_function_index_y = self._BEM_manager.get_basis_function_index(self._Q_Dirichlet, jj)

                            local_charge_I = self._BEM_manager.proj_from_local_panel_index_to_local_charge_index(
                                Q_=self._Q_Neumann,
                                local_panel_index=local_I,
                                basis_func_index=basis_function_index_x,
                                panel_type=panel_type_i
                            )
                            local_charge_J = self._BEM_manager.proj_from_local_panel_index_to_local_charge_index(
                                Q_=self._Q_Dirichlet,
                                local_panel_index=local_J,
                                basis_func_index=basis_function_index_y,
                                panel_type=panel_type_j
                            )

                            integrand = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
                            for gauss_number in range(GaussQR4):
                                iii = gauss_number // GaussQR2
                                jjj = gauss_number % GaussQR2
                                rands = ti.Vector([iii // self._GaussQR, iii % self._GaussQR, jjj // self._GaussQR, jjj % self._GaussQR], ti.i32)

                                integrand += self.integrate_on_two_panels(
                                    rands=rands,
                                    scope_type=scope_type, k=k, sqrt_n=sqrt_n,
                                    panel_x=global_i, panel_y=global_j,
                                    basis_function_index_x=basis_function_index_x, basis_function_index_y=basis_function_index_y,
                                    panels_relation=panels_relation
                                )
                            
                            if local_charge_I >= 0 and local_charge_J >= 0:
                                self._Kmat[local_charge_I, local_charge_J] += integrand
    
    @ti.kernel
    def apply_K_dot_D_boundary(self, scope_type: int, k: float, sqrt_n: float, multiplier: float):
        """
        If you are applying Dirichlet boundary on vertices,
        You need to solve a linear system equations to get Neumann panels,
        Usually, a linear system equations requires a rhs, which

        rhs_{Dirichlet part} = Mg / 2 + Kg - Vf

        In this function, a [Kg] will be computed,
        where [g] is the input argument [vert_boundary] where a Dirichlet bounary is applied on vertices
        and [K] is our own BIO matrix [self._Kmat]
        """
        basis_func_num_Neumann = self._BEM_manager.get_num_of_basis_functions_from_Q(self._Q_Neumann)
        basis_func_num_Dirichlet = self._BEM_manager.get_num_of_basis_functions_from_Q(self._Q_Dirichlet)

        GaussQR2 = self._GaussQR * self._GaussQR
        GaussQR4 = GaussQR2 * GaussQR2

        for local_I in range(self.num_of_panels_Neumann):
            for local_J in range(self.num_of_panels_Neumann):
                global_i = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_I)
                global_j = self._BEM_manager.map_local_Neumann_index_to_panel_index(local_J)

                panel_type_i = self._BEM_manager.get_panel_type(global_i)
                if panel_type_i == int(CellFluxType.BOTH_TOBESOLVED):
                    panel_type_i = int(CellFluxType.NEUMANN_TOBESOLVED)

                panels_relation = self._BEM_manager.get_panels_relation(global_i, global_j)
                for ii in range(basis_func_num_Neumann):
                    for jj in range(basis_func_num_Dirichlet):
                        basis_function_index_x = self._BEM_manager.get_basis_function_index(self._Q_Neumann, ii)
                        basis_function_index_y = self._BEM_manager.get_basis_function_index(self._Q_Dirichlet, jj)

                        local_charge_I = self._BEM_manager.proj_from_local_panel_index_to_local_charge_index(
                            Q_=self._Q_Neumann,
                            local_panel_index=local_I,
                            basis_func_index=basis_function_index_x,
                            panel_type=panel_type_i
                        )

                        integrand = ti.Vector([0.0 for i in range(self._n)], self._ti_dtype)
                        for gauss_number in range(GaussQR4):
                            iii = gauss_number // GaussQR2
                            jjj = gauss_number % GaussQR2
                            rands = ti.Vector([iii // self._GaussQR, iii % self._GaussQR, jjj // self._GaussQR, jjj % self._GaussQR], ti.i32)
                            integrand += self.integrate_on_two_panels(
                                rands=rands,
                                scope_type=scope_type, k=k, sqrt_n=sqrt_n,
                                panel_x=global_i, panel_y=global_j,
                                basis_function_index_x=basis_function_index_x, basis_function_index_y=basis_function_index_y,
                                panels_relation=panels_relation
                            )

                        gy = self._BEM_manager.rhs_constructor.get_Dirichlet_boundary(global_j, basis_function_index_y)

                        Neumann_offset_i = self._BEM_manager.get_Neumann_offset_i()
                        if local_charge_I >= 0:
                            if ti.static(self._n == 1):
                                self._BEM_manager.rhs_constructor.get_rhs_vec()[local_charge_I + Neumann_offset_i] += multiplier * integrand * gy
                            elif ti.static(self._n == 2):
                                self._BEM_manager.rhs_constructor.get_rhs_vec()[local_charge_I + Neumann_offset_i] += multiplier * ti.math.cmul(integrand, ti.math.cconj(gy))
