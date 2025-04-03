import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")


class Poisson:
    def __init__(self, M, f, u_exact, d_exact=None, dd_exact=None, N_fine=100):
        self.M = M
        self.N = 2 * M + 1 
        self.N_fine = N_fine
        self.f = f
        self.ex = u_exact
        self.d_ex = d_exact if d_exact else lambda x: np.gradient(self.ex(x), x)
        self.dd_ex = dd_exact if dd_exact else lambda x: np.gradient(self.d_ex(x), x)

        self.xi_b = np.array([0.0, 0.5, 1.0])  # reference element
        self.nodes = np.linspace(0, 1, self.N) 
        self.elements = self.nodes[::2]  # nodes of the elements
        self.uh = None
        self.xh = None
        self.A = None
        self.b = None
        
        self.free = np.arange(1, self.N - 1)  # free nodes (Dirichlet BC: u(0)=u(1)=0)
        self.u = np.zeros(self.N)

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

    @staticmethod
    def ddphi(i, xi):
        """Second derivative of reference shape function wrt xi in [0,1]."""
        if i == 0:
            return 4.0
        elif i == 1:
            return -8.0
        elif i == 2:
            return 4.0
        else:
            raise ValueError("Local shape index must be 0,1,2.")

    @staticmethod
    def dddphi(i, xi):
        """Third derivative of reference shape function wrt xi in [0,1]."""
        if i == 0:
            return 0.0
        elif i == 1:
            return 0.0
        elif i == 2:
            return 0.0
        else:
            raise ValueError("Local shape index must be 0,1,2.")

    @staticmethod
    def theta(k, a):
        return 2 * k + a

    @staticmethod
    def localStiffnessMatrix(hk):
        return np.array([[7.0, -8.0, 1.0], [-8.0, 16.0, -8.0], [1.0, -8.0, 7.0]]) / (
            3 * hk
        )  # local stiffness matrix

    @staticmethod
    def localLoadVector(hk, f, i):
        w_smp = np.array([1.0, 4.0, 1.0])
        return hk * f * w_smp[i] / 6.0  # local load vector

    def global_assembly(self):
        self.A = np.zeros((self.N, self.N), dtype=float)
        self.b = np.zeros(self.N, dtype=float)
        for k in range(self.M):
            idx = np.array([2 * k, 2 * k + 1, 2 * k + 2])
            xL = self.nodes[idx[0]]
            xR = self.nodes[idx[-1]]
            # print(f"idx: {idx}, xL: {xL}, xR: {xR}")
            hk = xR - xL
            A_loc = self.localStiffnessMatrix(hk)
            b_loc = np.zeros(3)

            for xi in self.xi_b:
                x = xL + hk * xi
                for i in range(3):
                    f = self.f(x) * self.phi(i, xi)
                    b_loc[i] += self.localLoadVector(hk, f, i)
            for a in range(3):
                self.A[idx[a], idx] += A_loc[a, :]
                self.b[idx[a]] += b_loc[a]
        A_int = self.A[self.free][:, self.free]
        b_int = self.b[self.free]
        return A_int, b_int
    
    def solve(self, N=100):
        A_int, b_int = self.global_assembly()
        self.u[self.free] = np.linalg.solve(A_int, b_int)
        
        xh, uh = self.interpolate(self.nodes[::2], self.u, N=N)
        self.uh = uh
        self.xh = xh
        return xh, uh

    def interpolate(self, elements, Uh, N=100):
        M = len(elements) - 1
        U = np.zeros(N * M)
        x_fine = np.linspace(0, 1, N)  # interpolation points
        domain = np.zeros(N * M)
        for k in range(M):
            a = elements[k]
            b = elements[k + 1]
            y = np.zeros_like(x_fine)
            Kk = a + x_fine * (b - a)
            for i in range(3):
                y += self.phi(i, x_fine) * Uh[2*k + i]
            U[N * k : N * k + N] = y
            domain[N * k : N * k + N] = Kk

        mask = np.zeros_like(domain, dtype=bool)
        mask[np.unique(domain, return_index=True)[1]] = True
        return domain[mask], U[mask]

    def error(self, norm="L2"):
        if self.uh is None and self.xh is None:
            uh, xh = self.solve()
        else:
            uh, xh = self.uh, self.xh
        u = self.ex(xh)
        e = u - uh
        if norm == "L2":
            return np.sqrt(simpson(e**2, xh))
        elif norm == "Linf":
            return np.max(np.abs(uh - u))
        elif norm == "H1":
            grad_u = np.gradient(uh, xh)
            grad_exact = np.gradient(u, xh)
            return np.sqrt(simpson(e**2, xh) + simpson((grad_u - grad_exact)**2, xh))

    def get_solution(self):
        if self.uh is None or self.xh is None:
            self.solve()
        return self.xh, self.uh

    def get_coefficients(self):
        if self.uh is None:
            self.solve()
        return self.u

    def get_exact_solution(self, x):
        return self.ex(x)

    def get_stiffness_matrix(self):
        if self.A is None:
            self.global_assembly()
        return self.A

    def get_load_vector(self):
        if self.b is None:
            self.global_assembly()
        return self.b




# ------------------------------------------
# *********** Utility Functions ************
# ------------------------------------------
def compute_convergence_data(f, u_ex, d_ex, dd_ex, M_list=None):
    """Compute convergence data across multiple mesh sizes"""
    if M_list is None:
        M_list = np.array([10, 20, 40, 80, 160, 320])
    
    h_list = 1 / M_list
    err_L2 = []
    err_Linf = []
    err_H1 = []
    solvers = []

    for M in M_list:
        solver = Poisson(M, f, u_ex, d_ex, dd_ex)
        solver.solve()
        err_L2.append(solver.error(norm="L2"))
        err_H1.append(solver.error(norm="H1"))
        err_Linf.append(solver.error(norm="Linf"))
        solvers.append(solver)
        
    # Compute convergence rates
    p_L2 = np.polyfit(np.log(h_list), np.log(err_L2), 1)[0]
    p_H1 = np.polyfit(np.log(h_list), np.log(err_H1), 1)[0]
    p_Linf = np.polyfit(np.log(h_list), np.log(err_Linf), 1)[0]
    
    return {
        'h_list': np.array(h_list),
        'M_list': M_list,
        'err_L2': np.array(err_L2),
        'err_H1': np.array(err_H1),
        'err_Linf': np.array(err_Linf),
        'p_L2': p_L2,
        'p_H1': p_H1,
        'p_Linf': p_Linf,
        'solvers': solvers
    }

def plot(f, u_ex, d_ex, dd_ex, name="test", savefig=False, M_list=None):
    """Plot FEM solution and convergence results"""
    data = compute_convergence_data(f, u_ex, d_ex, dd_ex, M_list)
    last_solver = data['solvers'][-1]
    
    fig, axs = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]}, dpi=100
    )
    
    # Plot exact solution and FEM solution
    xh, uh = last_solver.get_solution()
    u = last_solver.get_exact_solution(xh)
    
    axs[0].plot(xh, u, "r-", linewidth=2, label="$u(x)$")
    axs[0].plot(xh, uh, "bo-", markersize=5, label="$u_h$")
    axs[0].set_title("FEM vs Exact Solution")
    axs[0].set_xlabel("$x$")
    axs[0].set_ylabel("$u(x)$")
    axs[0].legend()
    axs[0].grid(True)

    # Plot convergence
    h_list = data['h_list']
    err_L2 = data['err_L2']
    err_H1 = data['err_H1']
    err_Linf = data['err_Linf']
    
    axs[1].loglog(
        h_list,
        err_L2,
        "bo-",
        linewidth=2,
        markersize=8,
        label=f"$\\|e_h\\|_{{L^2}} = \\mathcal{{O}}(h^{{{data['p_L2']:.2f}}})$",
    )
    axs[1].loglog(
        h_list,
        err_H1,
        "mo-",
        linewidth=2,
        markersize=8,
        label=f"$\\|e_h\\|_{{H^1}} = \\mathcal{{O}}(h^{{{data['p_H1']:.2f}}})$",
    )
    axs[1].loglog(
        h_list,
        err_Linf,
        "go-",
        linewidth=2,
        markersize=8,
        label=f"$\\|e_h\\|_{{L^\\infty}} = \\mathcal{{O}}(h^{{{data['p_Linf']:.2f}}})$",
    )

    # Reference rates
    axs[1].loglog(
        h_list,
        err_L2[0] * (h_list / h_list[0]) ** 3,
        "r--",
        linewidth=2,
        label="$\\mathcal{O}(h^3)$",
    )
    axs[1].loglog(
        h_list,
        err_L2[0] * (h_list / h_list[0]) ** 2,
        "g--",
        linewidth=2,
        label="$\\mathcal{O}(h^2)$",
    )
    axs[1].loglog(
        h_list,
        err_L2[0] * (h_list / h_list[0]) ** 1,
        "c--",
        linewidth=2,
        label="$\\mathcal{O}(h^1)$",
    )
    axs[1].set_xlabel("Mesh size $h$")
    axs[1].set_ylabel("$\\|e\\|$")
    axs[1].set_title("Convergence Plot")
    axs[1].grid(True, which="both", ls="--", alpha=0.7)
    axs[1].legend()

    plt.tight_layout()
    if savefig:
        plt.savefig(f"figures/fem_plot_convergence_{name}_M{last_solver.M}.png", dpi=100)
    plt.show()

def plot_stiffness_matrix_and_load_vector(solver: Poisson, name="test", savefig=False):
    """Plot stiffness matrix and load vector"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    A = solver.get_stiffness_matrix()
    b = solver.get_load_vector()

    # Plot stiffness matrix
    ims = ax[0].imshow(A, cmap="inferno")
    ax[0].set_title("Stiffness Matrix: $\\mathbf{A}$")
    ax[0].set_xlabel("Node Index $j$")
    ax[0].set_ylabel("Node Index $i$")
    cb = plt.colorbar(ims, ax=ax[0], orientation="vertical", pad=0.02)
    cb.set_label("Value")
    ticks = np.linspace(np.min(A), np.max(A), 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{tick:.2f}" for tick in ticks])

    # Plot load vector
    N = solver.N
    nodes = np.arange(N)
    ax[1].bar(nodes, b, color="blue", alpha=0.7)
    ax[1].set_title("Load Vector $\\mathbf{b}$")
    ax[1].set_xlabel("Node Index $i = \\theta(k, j)$")
    ax[1].set_ylabel("Value")
    plt.tight_layout()
    if savefig:
        plt.savefig(f"figures/stiffness_load_{name}_M{solver.M}.png", dpi=100)
    plt.show()

def convergence_test(solver, M_list=None, N_list=None):
    """Run comprehensive convergence test"""
    if M_list is None:
        M_list = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    
    h_list = 1 / M_list
    err_L2 = []
    err_inf = []
    err_H1 = []
    rate_L2 = []
    rate_inf = []
    rate_H1 = []
    
    if N_list is None:
        N_list = [2 * M + 1 for M in M_list]
    
    # Compute errors for each mesh size
    for m, N_fine in zip(M_list, N_list):
        Sol_ = Poisson(m, solver.f, solver.ex, solver.d_ex, solver.dd_ex, N_fine=N_fine)
        Sol_.solve()
        err_L2.append(Sol_.error(norm="L2"))
        err_H1.append(Sol_.error(norm="H1"))
        err_inf.append(Sol_.error(norm="Linf"))
    
    # Convert to numpy arrays
    err_L2 = np.array(err_L2)
    err_inf = np.array(err_inf)
    err_H1 = np.array(err_H1)
    
    # Compute convergence rates between consecutive mesh refinements
    for i in range(1, len(h_list)):
        rate_L2.append(np.log(err_L2[i-1]/err_L2[i]) / np.log(h_list[i-1]/h_list[i]))
        rate_inf.append(np.log(err_inf[i-1]/err_inf[i]) / np.log(h_list[i-1]/h_list[i]))
        rate_H1.append(np.log(err_H1[i-1]/err_H1[i]) / np.log(h_list[i-1]/h_list[i]))
    
    # Compute average rates
    avg_rate_L2 = np.mean(rate_L2)
    avg_rate_inf = np.mean(rate_inf)
    avg_rate_H1 = np.mean(rate_H1)
    
    # Also compute overall convergence rates using polyfit for comparison
    p_L2 = np.polyfit(np.log(h_list), np.log(err_L2), 1)[0]
    p_inf = np.polyfit(np.log(h_list), np.log(err_inf), 1)[0]
    p_H1 = np.polyfit(np.log(h_list), np.log(err_H1), 1)[0]
    
    return (h_list, (p_L2, p_inf, p_H1), (err_L2, err_inf, err_H1), 
            (rate_L2, rate_inf, rate_H1), (avg_rate_L2, avg_rate_inf, avg_rate_H1))
    
def print_convergence_table(solver, M_list=None):
    """Print detailed convergence analysis table"""
    hs, (p_L2, p_inf, p_H1), (errors_L2, errors_inf, errors_H1), \
    (rates_L2, rates_inf, rates_H1), (avg_L2, avg_inf, avg_H1) = convergence_test(solver, M_list)
    
    # Table formatting
    headers = ["M", "h", "L² Error", "Rate", "L∞ Error", "Rate", "H¹ Error", "Rate"]
    col_widths = [8, 12, 15, 8, 15, 8, 15, 8]
    header_fmt = " | ".join(["{:" + str(w) + "s}" for w in col_widths])
    row_fmt = " | ".join(["{:" + str(w) + "}" for w in col_widths])
    separator = "-+-".join(["-" * w for w in col_widths])
    title_width = sum(col_widths) + 3 * (len(col_widths) - 1)
    
    print("\n" + "=" * title_width)
    print("CONVERGENCE ANALYSIS".center(title_width))
    print("=" * title_width)
    print(header_fmt.format(*headers))
    print(separator)

    # Print first row without rates
    M_val = int(1.0 / hs[0])
    row = [
        f"{M_val:d}",
        f"{hs[0]:.2e}",
        f"{errors_L2[0]:.2e}",
        "--",
        f"{errors_inf[0]:.2e}",
        "--",
        f"{errors_H1[0]:.2e}",
        "--"
    ]
    print(row_fmt.format(*row))

    # Print remaining rows with rates
    for i in range(1, len(hs)):
        M_val = int(1.0 / hs[i])
        row = [
            f"{M_val:d}",
            f"{hs[i]:.2e}",
            f"{errors_L2[i]:.2e}",
            f"{rates_L2[i-1]:.2f}",
            f"{errors_inf[i]:.2e}",
            f"{rates_inf[i-1]:.2f}",
            f"{errors_H1[i]:.2e}",
            f"{rates_H1[i-1]:.2f}"
        ]
        print(row_fmt.format(*row))

    print(separator)
    print("\nAverage convergence rates:")
    print(f"L²: {avg_L2:.2f}    L∞: {avg_inf:.2f}    H¹: {avg_H1:.2f}")
    print("\nOverall convergence rates (from polyfit):")
    print(f"L²: {p_L2:.2f}    L∞: {p_inf:.2f}    H¹: {p_H1:.2f}")
    print("=" * title_width + "\n")


def plot_convergence(solver, name="test", label="test", savefig=False, M_list=None, N_list=None):
    """Plot convergence analysis with improved visualization
    
    Parameters:
        solver: Poisson solver instance
        name: Name for saving figure
        label: Title label for the plot
        savefig: Whether to save figure
        M_list: List of mesh sizes
        N_list: List of fine mesh sizes for error computation
    """
    # Get convergence data
    hs, (p_L2, p_inf, p_H1), (errors_L2, errors_inf, errors_H1), \
    (rates_L2, rates_inf, rates_H1), _ = convergence_test(solver, M_list, N_list)
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Top plot: Solution comparison
    ax_s = fig.add_subplot(gs[0, :])
    x_fine = np.linspace(0, 1, 200)
    u_exact = solver.get_exact_solution(x_fine)
    u_nodes = solver.get_coefficients()
    x_nodes = solver.nodes

        
    # Plot exact and numerical solutions
    ax_s.plot(x_fine, u_exact, "r-", linewidth=2, label="Exact")
    ax_s.plot(x_nodes, u_nodes, "bo", markersize=5, label="FEM", alpha=0.7)
    ax_s.set_title(f"FEM vs Exact Solution\n{label}")
    ax_s.set_xlabel("$x$")
    ax_s.set_ylabel("$u(x)$")
    ax_s.legend()
    ax_s.grid(True)

    # Bottom left: Convergence rates
    ax_c = fig.add_subplot(gs[1, 0])
    
    # Plot error curves
    ax_c.loglog(hs, errors_L2, "bo-", linewidth=2, label=f"$L^2: O(h^{{{p_L2:.2f}}})$")
    ax_c.loglog(hs, errors_inf, "go-", linewidth=2, label=f"$L^\\infty: O(h^{{{p_inf:.2f}}})$")
    ax_c.loglog(hs, errors_H1, "mo-", linewidth=2, label=f"$H^1: O(h^{{{p_H1:.2f}}})$")
    
    # Add reference lines
    h_ref = np.array([hs[0], hs[-1]])
    ax_c.loglog(h_ref, errors_L2[0]*(h_ref/hs[0])**3, "k--", alpha=0.5, label="$O(h^3)$")
    ax_c.loglog(h_ref, errors_L2[0]*(h_ref/hs[0])**2, "k:", alpha=0.5, label="$O(h^2)$")
    
    ax_c.set_xlabel("Mesh size $h$")
    ax_c.set_ylabel("Error $\\|e_h\\|$")
    ax_c.grid(True, which='both', ls='-', alpha=0.6)
    ax_c.legend(loc='lower right')
    ax_c.set_title("Convergence Rates")

    # Bottom right: Error evolution
    ax_e = fig.add_subplot(gs[1, 1])
    refinements = np.arange(len(hs))
    
    # Plot error evolution
    ax_e.semilogy(refinements, errors_L2, "bo-", label="$L^2$")
    ax_e.semilogy(refinements, errors_inf, "go-", label="$L^\\infty$")
    ax_e.semilogy(refinements, errors_H1, "mo-", label="$H^1$")
    
    # Add rate annotations
    for i in range(1, len(refinements)):
        y_pos = np.sqrt(errors_L2[i] * errors_L2[i-1])
        ax_e.text(i-0.5, y_pos, f"{rates_L2[i-1]:.2f}", 
                 horizontalalignment='center', color='blue', alpha=0.7)
    
    ax_e.set_xlabel("Refinement level")
    ax_e.set_ylabel("Error")
    ax_e.grid(True)
    ax_e.legend()
    ax_e.set_title("Error Evolution with Rates")

    plt.tight_layout()
    if savefig:
        plt.savefig(f"figures/convergence_{name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    