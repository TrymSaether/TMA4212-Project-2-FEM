import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")


class FEMPoissonSolver:
    def __init__(self, M, f, exact, du_exact, ddu_exact):
        self.M = M
        self.f = f
        self.exact = exact
        self.h = 1.0 / M
        self.N = 2 * M + 1
        self.nodes = np.linspace(0, 1, self.N)  # mesh points
        self.A_ref = np.array(
            [[7 / 3, -8 / 3, 1 / 3], [-8 / 3, 16 / 3, -8 / 3], [1 / 3, -8 / 3, 7 / 3]]
        )  # local stiffness matrix
        self.xi_quad = np.array([0.0, 0.5, 1.0])  # ref elements
        self.w_quad = np.array([1 / 6, 4 / 6, 1 / 6])  # weights for Simpson's rule
        self.A = np.zeros((self.N, self.N))
        self.F = np.zeros(self.N)
        self.du_exact = du_exact
        self.ddu_exact = ddu_exact

    @staticmethod
    def phi(i, xi):
        if i == 0:
            return 2.0 * xi**2 - 3.0 * xi + 1.0
        elif i == 1:
            return -4.0 * xi**2 + 4.0 * xi
        elif i == 2:
            return 2.0 * xi**2 - xi

    @staticmethod
    def dphi(i, xi):
        """Derivative of reference shape function wrt xi in [0,1]."""
        if i == 0:
            return 4.0 * xi - 3.0
        elif i == 1:
            return -8.0 * xi + 4.0
        elif i == 2:
            return 4.0 * xi - 1.0
        else:
            raise ValueError("Local shape index must be 0,1,2.")

    def ddphi(self, i, xi):
        """Second derivative of reference shape function wrt xi in [0,1]."""
        if i == 0:
            return 4.0
        elif i == 1:
            return -8.0
        elif i == 2:
            return 4.0
        else:
            raise ValueError("Local shape index must be 0,1,2.")

    def dddphi(self, i, xi):
        """Third derivative of reference shape function wrt xi in [0,1]."""
        if i == 0:
            return 0.0
        elif i == 1:
            return 0.0
        elif i == 2:
            return 0.0
        else:
            raise ValueError("Local shape index must be 0,1,2.")

    def assemble(self):
        for k in range(self.M):
            indices = np.array([2 * k, 2 * k + 1, 2 * k + 2])
            h_k = self.nodes[indices[-1]] - self.nodes[indices[0]]
            A_loc = (1 / h_k) * self.A_ref  # local stiffness matrix
            b_loc = np.zeros(3)
            x_left = self.nodes[indices[0]]
            # Compute local load vector using Simpson’s rule
            for q, xi in enumerate(self.xi_quad):
                x_q = x_left + h_k * xi
                for i in range(3):
                    b_loc[i] += self.w_quad[q] * self.f(x_q) * self.phi(i, xi)
            b_loc *= h_k
            # Assemble local contributions
            for a in range(3):
                self.A[indices[a], indices] += A_loc[a, :]
                self.F[indices[a]] += b_loc[a]
        # Dirichlet BC: u(0)=0 and u(1)=0.
        free = np.arange(1, self.N - 1)
        self.free = free
        self.A_reduced = self.A[free][:, free]
        self.F_reduced = self.F[free]

    def solve(self):
        self.u_free = np.linalg.solve(self.A_reduced, self.F_reduced)
        self.u = np.zeros(self.N)
        self.u[self.free] = self.u_free
        return self.u

    def eval_solution(self, x):
        """
        Evaluate the finite element solution at point x in [0,1].
        We'll find which element x belongs to, then do an affine map
        to reference coords, and sum local shape functions times dofs.
        """
        if x <= 0:
            return self.u[0]
        if x >= 1:
            return self.u[-1]

        # identify which element: find k s.t. x in [ x_{2k}, x_{2k+2} ]
        # a naive approach:
        kguess = int(x / self.h)  # approximate element index
        # clamp to valid range
        if kguess >= self.M:
            kguess = self.M - 1

        xL = self.nodes[2 * kguess]  # left node
        xR = self.nodes[2 * kguess + 2]  # right node
        if not (xL <= x <= xR):
            # refine search if needed, or do a direct search
            # for simplicity assume this guess is correct enough
            pass

        idx_local = [2 * kguess, 2 * kguess + 1, 2 * kguess + 2]
        h_k = xR - xL
        xi = (x - xL) / h_k  # map x -> xi in [0,1]

        # sum over local shape functions
        uh_val = 0.0
        for i_local in range(3):
            gi = idx_local[i_local]  # global index
            uh_val += self.u[gi] * self.phi(i_local, xi)
        return uh_val

    def eval_solution_derivative(self, x):
        """
        Evaluate derivative of the FE solution at point x in [0,1].
        The derivative wrt x is (1/h_k)* dphi/dxi for each local shape.
        """
        if x <= 0 or x >= 1:
            # derivative at boundary can be found by local element approach:
            # but let's just handle the same as interior
            pass

        kguess = int(x / self.h)
        if kguess >= self.M:
            kguess = self.M - 1

        xL = self.nodes[2 * kguess]
        xR = self.nodes[2 * kguess + 2]
        idx_local = [2 * kguess, 2 * kguess + 1, 2 * kguess + 2]
        h_k = xR - xL

        xi = (x - xL) / h_k

        du_val = 0.0
        for i_local in range(3):
            gi = idx_local[i_local]
            # chain rule: d/dx phi = (1/h_k)* d/dxi phi
            du_val += self.u[gi] * (1.0 / h_k) * self.dphi(i_local, xi)
        return du_val

    def compute_L2_norm(self):
        """
        Compute the L2 norm of the FE solution, i.e. sqrt( int_0^1 (u_h(x))^2 dx ).
        We can do an elementwise Simpson rule again, re-using self.xi_quad, self.w_quad.
        """
        sq_sum = 0.0
        for k in range(self.M):
            xL = self.nodes[2 * k]
            xR = self.nodes[2 * k + 2]
            h_k = xR - xL
            for q, xi in enumerate(self.xi_quad):
                x_q = xL + h_k * xi
                wq = self.w_quad[q]
                u_val = self.eval_solution(x_q)
                u_exact_val = self.exact(x_q)
                sq_sum += wq * ((u_exact_val - u_val) ** 2)
            sq_sum *= h_k
        return np.sqrt(sq_sum)

    def compute_H1_norm(self):
        """
        Compute full H^1 norm: sqrt( int_0^1 (u_h(x))^2 + (u_h'(x))^2 dx ).
        """
        sq_sum = 0.0
        for k in range(self.M):
            xL = self.nodes[2 * k]
            xR = self.nodes[2 * k + 2]
            h_k = xR - xL
            for q, xi in enumerate(self.xi_quad):
                x_q = xL + h_k * xi
                wq = self.w_quad[q]
                u_val = self.eval_solution(x_q)
                du_val = self.eval_solution_derivative(x_q)
                du_exact_val = self.du_exact(x_q)
                u_exact_val = self.exact(x_q)
                sq_sum += wq * ((u_exact_val - u_val) ** 2 + (du_exact_val - du_val) ** 2)
            sq_sum *= h_k
        return np.sqrt(sq_sum)

    def compute_H2_norm(self):
        """
        Compute full H^2
        norm: sqrt( int_0^1 (u_h(x))^2 + (u_h'(x))^2 + (u_h''(x))^2 dx ).
        """
        sq_sum = 0.0
        for k in range(self.M):
            xL = self.nodes[2 * k]
            xR = self.nodes[2 * k + 2]
            h_k = xR - xL
            for q, xi in enumerate(self.xi_quad):
                x_q = xL + h_k * xi
                wq = self.w_quad[q]
                u_val = self.eval_solution(x_q)
                u_exact_val = self.exact(x_q)
                du_val = self.eval_solution_derivative(x_q)
                du_exact_val = self.du_exact(x_q)
                ddu_val = self.ddphi(1, xi)
                ddu_exact_val = self.ddu_exact(x_q)
                sq_sum += wq * (
                    (u_exact_val - u_val) ** 2
                    + (du_exact_val - du_val) ** 2
                    + (ddu_exact_val - ddu_val) ** 2
                )
            sq_sum *= h_k
        return np.sqrt(sq_sum)

    def compute_H3_norm(self):
        return self.compute_H2_norm()

    def plot_solution(self, fine_mesh=200, name="test", savefig=False):
        x_fine = np.linspace(0, 1, fine_mesh)
        u_exact = self.exact(x_fine)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

        ax.plot(x_fine, u_exact, "r-", linewidth=2, label="Exact Solution")
        ax.plot(self.nodes, self.u, "bo-", markersize=5, label="FEM Solution")
        ax.set_title("FEM vs Exact Solution")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u(x)$")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        if savefig:
            plt.savefig(f"figures/fem_plot_{name}_M{self.M}.png", dpi=100)
        plt.show()

    def plot(self, fine_mesh=200, name="test", savefig=False):
        fig, axs = plt.subplots(
            1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]}, dpi=150
        )
        
        x_fine = np.linspace(0, 1, fine_mesh)
        u_exact = np.array([self.exact(x) for x in x_fine])
        # Plot exact solution and FEM solution
        axs[0].plot(x_fine, u_exact, "r-", linewidth=2, label="Exact Solution")
        axs[0].plot(self.nodes, self.u, "bo-", markersize=5, label="FEM Solution")
        axs[0].set_title("FEM vs Exact Solution")
        axs[0].set_xlabel("$x$")
        axs[0].set_ylabel("$u(x)$")
        axs[0].legend()
        axs[0].grid(True)

        M_vals = [10, 20, 40, 80, 160]
        errors = []
        hs = []
        for M in M_vals:
            temp_solver = FEMPoissonSolver(M, self.f, self.exact, self.du_exact, self.ddu_exact)
            temp_solver.assemble()
            temp_solver.solve()
            errors.append(temp_solver.compute_L2_norm())
            hs.append(1.0 / M)

        hs = np.array(hs)
        errors = np.array(errors)
        p = np.polyfit(np.log(hs), np.log(errors), 1)
        ref_hs = hs / hs[0]

        # Plot convergence
        axs[1].loglog(
            hs,
            errors,
            "bo-",
            linewidth=2,
            markersize=8,
            label=f"$\\|e_h\\|_{{L^2}} = \\mathcal{{O}}(h^{{{p[0]:.2f}}})$",
        )
        axs[1].loglog(
            hs,
            errors[0] * (ref_hs) ** 3,
            "r--",
            linewidth=2,
            label="$\\mathcal{O}(h^3)$",
        )
        axs[1].loglog(
            hs,
            errors[0] * (ref_hs) ** 2,
            "g--",
            linewidth=2,
            label="$\\mathcal{O}(h^2)$",
        )
        axs[1].set_xlabel("Mesh size $h$")
        axs[1].set_ylabel("$\\|e\\|_{\\mathrm{L}^2}$")
        axs[1].grid(True, which="both", ls="--", alpha=0.7)
        axs[1].legend()
        axs[1].set_title("Convergence Plot")

        plt.tight_layout()
        if savefig:
            plt.savefig(f"figures/fem_plot_convergence_{name}_M{self.M}.png", dpi=200)
        plt.show()

    def plot_stiffness_matrix_and_load_vector(self, name="test", savefig=False):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

        # Plot stiffness matrix
        ims = ax[0].imshow(self.A, cmap="hot", aspect="auto")
        ax[0].set_title("Stiffness Matrix $\\mathbf{A}$")
        ax[0].set_xlabel("Node Index $j$")
        ax[0].set_ylabel("Node Index $i$")
        cb = plt.colorbar(ims, ax=ax[0], orientation="vertical", pad=0.02)
        cb.set_label("Value")
        ticks = np.linspace(np.min(self.A), np.max(self.A), 5)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{tick:.2f}" for tick in ticks])

        # Plot load vector
        bax = ax[1].bar(np.arange(self.N), self.F, color="blue", alpha=0.7)
        ax[1].set_title("Load Vector $\\mathbf{F}$")
        ax[1].set_xlabel("Node Index $i = \\theta(k, j)$")
        ax[1].set_ylabel("Value")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"figures/stiffness_load_{name}_M{self.M}.png", dpi=200)
        plt.show()
    def convergence_test(self):
        M_vals = [4, 8, 16, 32, 64, 128, 256, 512]
        errors_L2 = []
        errors_H1 = []
        errors_H2 = []
        errors_H3 = []
        hs = []
        # Compute L2 error for each M
        for M in M_vals:
            temp_solver = FEMPoissonSolver(M, self.f, self.exact, self.du_exact, self.ddu_exact)
            temp_solver.assemble()
            temp_solver.solve()

            errors_L2.append(temp_solver.compute_L2_norm())
            errors_H1.append(temp_solver.compute_H1_norm())
            errors_H2.append(temp_solver.compute_H2_norm())
            hs.append(1.0 / M)

        hs = np.array(hs)
        errors_L2 = np.array(errors_L2)
        errors_H1 = np.array(errors_H1)
        errors_H2 = np.array(errors_H2)
        p_L2 = np.polyfit(np.log(hs), np.log(errors_L2), 1)[0]
        p_H1 = np.polyfit(np.log(hs), np.log(errors_H1), 1)[0]
        p_H2 = np.polyfit(np.log(hs), np.log(errors_H2), 1)[0]
        return (
            hs,
            (p_L2, p_H1, p_H2),
            (errors_L2, errors_H1, errors_H2),
        )

    def print_convergence_table(self):
        """
        Print a formatted table of convergence results.
        """
        hs, (p_L2, p_H1, p_H2), (errors_L2, errors_H1, errors_H2) = (
            self.convergence_test()
        )
        width = 120
        col_width = 12

        print("\n" + "Convergence Analysis".center(width))
        print("=" * width)
        headers = [
            "M",
            "h",
            "L2 Error",
            "Rate L2",
            "Ratio L2",
            "H1 Error",
            "Rate H1",
            "Ratio H1",
            "H2 Error",
            "Rate H2",
            "Ratio H2"
        ]
        print("".join(h.center(col_width) for h in headers))
        print("-" * width)

        # Since hs = 1/M, compute M as int(1/h)
        for i in range(len(hs)):
            M_val = int(1.0 / hs[i])
            if i == 0:
                rate_L2 = rate_H1 = rate_H2 = "-"
                ratio_L2 = ratio_H1 = ratio_H2 = "-"
            else:
                rate_L2 = f"{np.log(errors_L2[i-1]/errors_L2[i])/np.log(2):.2f}"
                rate_H1 = f"{np.log(errors_H1[i-1]/errors_H1[i])/np.log(2):.2f}"
                rate_H2 = f"{np.log(errors_H2[i-1]/errors_H2[i])/np.log(2):.2f}"
                ratio_L2 = f"{errors_L2[i-1]/errors_L2[i]:.2f}"
                ratio_H1 = f"{errors_H1[i-1]/errors_H1[i]:.2f}"
                ratio_H2 = f"{errors_H2[i-1]/errors_H2[i]:.2f}"

            row = [
                f"{M_val}",
                f"{hs[i]:.2e}",
                f"{errors_L2[i]:.2e}",
                rate_L2,
                ratio_L2,
                f"{errors_H1[i]:.2e}",
                rate_H1,
                ratio_H1,
                f"{errors_H2[i]:.2e}",
                rate_H2,
                ratio_H2
            ]
            print("".join(str(item).center(col_width) for item in row))

        print("=" * width)
        print(
            f"Overall convergence rates - L2: {p_L2:.2f}, H1: {p_H1:.2f}, H2: {p_H2:.2f}"
        )
        print(
            f"Final errors - L2: {errors_L2[-1]:.2e}, H1: {errors_H1[-1]:.2e}, H2: {errors_H2[-1]:.2e}"
        )
        print("=" * width + "\n")

    def plot_convergence(self, name="test", savefig=False):
        """
        Plot convergence graphs for various error norms.
        """
        hs, (p_L2, p_H1, p_H2), (errors_L2, errors_H1, errors_H2) = (
            self.convergence_test()
        )
        ref_hs = hs / hs[0]

        fig, ax = plt.subplots(
            1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]}, dpi=200
        )

        # Left panel: log-log plot of all error norms
        ax[0].loglog(
            hs,
            errors_L2,
            "bo-",
            linewidth=2,
            markersize=8,
            label=f"$\\|e_h\\|_{{L^2}} = \\mathcal{{O}}(h^{{{p_L2:.2f}}})$",
        )
        ax[0].loglog(
            hs,
            errors_H1,
            "mo-",
            linewidth=2,
            markersize=8,
            label=f"$\\|e_h\\|_{{H^1}} = \\mathcal{{O}}(h^{{{p_H1:.2f}}})$",
        )
        ax[0].loglog(
            hs,
            errors_H2,
            "go-",
            linewidth=2,
            markersize=8,
            label=f"$\\|e_h\\|_{{H^2}} = \\mathcal{{O}}(h^{{{p_H2:.2f}}})$",
        )

        ax[0].loglog(
            hs,
            errors_L2[0] * (ref_hs) ** 3,
            "r--",
            linewidth=2,
            label="$\\mathcal{O}(h^3)$",
            alpha=0.5,
        )
        ax[0].loglog(
            hs,
            errors_L2[0] * (ref_hs) ** 2,
            "g--",
            linewidth=2,
            label="$\\mathcal{O}(h^2)$",
            alpha=0.5,
        )
        ax[0].loglog(
            hs,
            errors_L2[0] * (ref_hs) ** 1,
            "c--",
            linewidth=2,
            label="$\\mathcal{O}(h)$",
            alpha=0.5,
        )

        ax[0].set_xlabel("Mesh size $h$")
        ax[0].set_ylabel("Error Norm")
        ax[0].grid(True, which="both", ls="--", alpha=0.7)
        ax[0].legend()
        ax[0].set_title("Convergence Plot")

        # Right panel: Plot individual error curves
        ax[1].plot(
            hs, errors_L2, "bo-", linewidth=2, markersize=8, label="$\\|e_h\\|_{L^2}$"
        )
        ax[1].plot(
            hs, errors_H1, "mo-", linewidth=2, markersize=8, label="$\\|e_h\\|_{H^1}$"
        )
        ax[1].plot(
            hs, errors_H2, "go-", linewidth=2, markersize=8, label="$\\|e_h\\|_{H^2}$"
        )
        ax[1].set_xlabel("Mesh size $h$")
        ax[1].set_ylabel("Error")
        ax[1].grid(True, which="both", ls="--", alpha=0.7)
        ax[1].legend()
        ax[1].set_title("Errors Plot")

        plt.tight_layout()
        if savefig:
            plt.savefig(f"figures/convergence_{name}_M{self.M}.png", dpi=200)
        plt.show()

    def get_exact_solution(self):
        """
        Evaluate the exact solution at the solver's global nodes, if available.
        """
        if self.exact is None:
            return None
        return np.array([self.exact(x) for x in self.nodes])
