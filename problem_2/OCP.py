import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from numpy.polynomial.legendre import leggauss

class FEMSolver:
    def __init__(self, n):
        """
        Initialize the 1D FEM solver with quadratic elements on [0,1].
        n : number of elements
        """
        self.n = n
        self.nodes = np.linspace(0, 1, 2*n + 1)
        self.elems = [(2*i, 2*i+1, 2*i+2) for i in range(n)]
        self.ndof = len(self.nodes)
        # interior node indices (Dirichlet BC: u(0)=u(1)=0)
        self.int_idx = slice(1, -1)
        # Gauss-Legendre quadrature with 3 points on [-1,1]
        self.qp, self.qw = leggauss(3)
        self.basis, self.dbasis = self._ref_basis()

    def _ref_basis(self):
        """Define quadratic Lagrange shape functions and their derivatives on [0,1]."""
        psi0 = lambda x: 2 * (x - 0.5) * (x - 1)
        psi1 = lambda x: 4 * x * (1 - x)
        psi2 = lambda x: 2 * x * (x - 0.5)
        dpsi0 = lambda x: 4 * x - 3
        dpsi1 = lambda x: 4 - 8 * x
        dpsi2 = lambda x: 4 * x - 1
        return [psi0, psi1, psi2], [dpsi0, dpsi1, dpsi2]

    def _assemble_local(self, h, use_mass=False):
        """
        Assemble the local 3x3 matrix for an element of length h.
        use_mass: if True, use the basis functions (for mass matrix);
                  if False, use the derivatives (for stiffness matrix).
        """
        loc = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                for qp, qw in zip(self.qp, self.qw):
                    # Map quadrature point from [-1,1] to [0,1]
                    xi = 0.5 * (qp + 1)
                    w = 0.5 * h * qw
                    if use_mass:
                        loc[i, j] += self.basis[i](xi) * self.basis[j](xi) * w
                    else:
                        loc[i, j] += self.dbasis[i](xi) * self.dbasis[j](xi) * w / h
        return loc

    def mass_matrix(self):
        """Assemble and return the global mass matrix M (with BC applied)."""
        M = lil_matrix((self.ndof, self.ndof))
        for el in self.elems:
            h = self.nodes[el[2]] - self.nodes[el[0]]
            locM = self._assemble_local(h, use_mass=True)
            for i_loc, i_glob in enumerate(el):
                for j_loc, j_glob in enumerate(el):
                    M[i_glob, j_glob] += locM[i_loc, j_loc]
        # Remove boundary rows/cols
        return csr_matrix(M[self.int_idx, :][:, self.int_idx])

    def assemble_system(self, f_func=None, use_mass=False):
        """
        Assemble the global matrix A and load vector b.
        f_func : source term (if provided, b is assembled).
        use_mass : if True, assemble mass matrix; otherwise, stiffness matrix.
        Returns: (A, b, x) where A and b have BC applied and x are the interior nodes.
        """
        A = lil_matrix((self.ndof, self.ndof))
        b = np.zeros(self.ndof)
        for el in self.elems:
            x0, _, x2 = self.nodes[el[0]], self.nodes[el[1]], self.nodes[el[2]]
            h = x2 - x0
            locA = self._assemble_local(h, use_mass=use_mass)
            locb = np.zeros(3)
            if f_func is not None:
                for i in range(3):
                    for qp, qw in zip(self.qp, self.qw):
                        xi = 0.5 * (qp + 1)
                        w = 0.5 * h * qw
                        x_phys = x0 + h * xi
                        locb[i] += f_func(x_phys) * self.basis[i](xi) * w
            for i_loc, i_glob in enumerate(el):
                for j_loc, j_glob in enumerate(el):
                    A[i_glob, j_glob] += locA[i_loc, j_loc]
                b[i_glob] += locb[i_loc]
        A = A[self.int_idx, :][:, self.int_idx]
        b = b[self.int_idx]
        return csr_matrix(A), b, self.nodes[self.int_idx]

    def solve_poisson(self, f_func):
        A, b, x = self.assemble_system(f_func, use_mass=False)
        u = spsolve(A, b)
        return x, u

    def solve_opt_control(self, alpha, y_d_func):
        """
        Solve the optimal control problem:
          min_{y,u in V_h}  (1/2)||y - ȳ_d||^2 + (α/2)||u||^2
          s.t. a(y, v) = (u, v) for all v in V_h.
        Here ȳ_d is the interpolation of y_d onto V_h.
        Returns: (x, y_h, u_h) with x the interior nodes.
        """
        K, _, x = self.assemble_system(f_func=None, use_mass=False)
        M = self.mass_matrix()
        Yd = y_d_func(x)
        n_int = len(x)
        A_blk = lil_matrix((2 * n_int, 2 * n_int))
        rhs = np.zeros(2 * n_int)
        A_blk[:n_int, :n_int] = M
        A_blk[:n_int, n_int:] = alpha * K
        A_blk[n_int:, :n_int] = K
        A_blk[n_int:, n_int:] = -M
        rhs[:n_int] = M @ Yd
        sol = spsolve(csr_matrix(A_blk), rhs)
        y = sol[:n_int]
        u = sol[n_int:]
        return x, y, u

def plot_opt_control_multi(n, alphas, yd):
    ocp = FEMSolver(n)
    
    x_fine = np.linspace(0, 1, 100)
    yd_fine = yd(x_fine)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=200, gridspec_kw={'width_ratios': [2.5, 1]})
    fig.subplots_adjust(wspace=0.3)
    
    # Use standard matplotlib colors (C0, C1, C3, etc.)
    colors = ['C0', 'C1', 'C3', 'C2', 'C4']  # Add more if needed
    
    axs[0].plot(x_fine, yd_fine, 'k--', linewidth=1.5, label='Desired $y_d$')
    
    for i, alpha in enumerate(alphas):
        color_idx = i % len(colors)
        x_int, y_int, u_int = ocp.solve_opt_control(alpha, yd)
        x_full = np.linspace(0, 1, len(ocp.nodes))
        y_full = np.zeros_like(ocp.nodes)
        y_full[1:-1] = y_int
        axs[0].plot(x_full, y_full, f"{colors[color_idx]}-", label=f'$\\alpha = 10^{{{int(np.log10(alpha))}}}$')
    
    axs[0].set_title(f'State Solutions')
    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('$y(x)$')
    axs[0].legend(loc='best')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    for i, alpha in enumerate(alphas):
        color_idx = i % len(colors)
        x_int, y_int, u_int = ocp.solve_opt_control(alpha, yd)
        axs[1].plot(x_int, u_int, f"{colors[color_idx]}o-", markersize=2, label=f'$\\alpha = 10^{{{int(np.log10(alpha))}}}$')
    axs[1].set_title('Control Solutions')
    axs[1].set_xlabel('$x$')
    axs[1].set_ylabel('$u(x)$')
    axs[1].legend(loc='best')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig

def print_convergence(solver, yd, alphas, n_vals):
    """
    Print H1 and L2 error, rate and order
    """
    print(f"Convergence Table for {yd.__name__}:")
    print(f"{'alpha':<10} {'n':<10} {'L2 Error':<15} {'L2 Rate':<15} {'L2 Order':<15} {'H1 Error':<15} {'H1 Rate':<15} {'H1 Order':<15}")
    print("="*111)
    
    for alpha in alphas:
        prev_L2_error = None
        prev_H1_error = None
        prev_h = None
        
        for n in n_vals:
            solver_temp = FEMSolver(n)
            x_int, y_int, u_int = solver_temp.solve_opt_control(alpha, yd)
            h = 1.0 / n
            
            L2_error = solver_temp.L2_error(y_int, yd(x_int), x_int)
            H1_error = solver_temp.H1_error(y_int, yd(x_int), x_int)
            
            if prev_L2_error is not None:
                L2_rate = np.log(prev_L2_error / L2_error) / np.log(prev_h / h)
                H1_rate = np.log(prev_H1_error / H1_error) / np.log(prev_h / h)
                L2_order = -np.log(L2_error) / np.log(h)
                H1_order = -np.log(H1_error) / np.log(h)
            else:
                L2_rate = np.nan
                H1_rate = np.nan
                L2_order = np.nan
                H1_order = np.nan
            
            print(f"{alpha:<10} {n:<10} {L2_error:<15.6f} {L2_rate:<15.6f} {L2_order:<15.6f} {H1_error:<15.6f} {H1_rate:<15.6f} {H1_order:<15.6f}")
            
            prev_L2_error = L2_error
            prev_H1_error = H1_error
            prev_h = h
            
    print("="*111)
