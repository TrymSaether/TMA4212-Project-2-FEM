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
        """
        Solve the Poisson problem -u'' = f with homogeneous Dirichlet BC.
        Returns: (x, u) with x the interior nodes and u the FEM solution.
        """
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
        # K: stiffness matrix from Poisson (using derivatives)
        K, _, x = self.assemble_system(f_func=None, use_mass=False)
        # M: mass matrix for L2 inner product
        M = self.mass_matrix()
        # Desired state at interior nodes
        Yd = y_d_func(x)
        n_int = len(x)
        # Build block system:
        #   [   M       αK   ] [ y ] = [ M*Yd ]
        #   [   K       -M   ] [ u ]   [   0  ]
        A_blk = lil_matrix((2 * n_int, 2 * n_int))
        rhs = np.zeros(2 * n_int)
        A_blk[:n_int, :n_int] = M
        A_blk[:n_int, n_int:] = alpha * K
        A_blk[n_int:, :n_int] = K
        A_blk[n_int:, n_int:] = -M
        rhs[:n_int] = M @ Yd
        sol = spsolve(csr_matrix(A_blk), rhs)
        return x, sol[:n_int], sol[n_int:]
    
    @staticmethod
    def L2_error(y_h, y_d, x):
        """Compute the L2 error between the computed and desired states."""
        return np.sqrt(np.mean((y_h - y_d)**2))

    @staticmethod
    def H1_error(y_h, y_d, x):
        """Compute the H1 error between the computed and desired states."""
        grad_y = np.gradient(y_h, x)
        grad_y_d = np.gradient(y_d, x)
        return np.sqrt(np.mean((y_h - y_d)**2) + np.mean((grad_y - grad_y_d)**2))

def plot_opt_control_multi(alphas, n_vals, yd, fine_mesh=200, savefig=False, name='opt_control_multi', error_norm='L2'):
    fig, axs = plt.subplots(1, 3, figsize=(14, 6), dpi=200, gridspec_kw={'width_ratios': [1, 1, 0.5]})
    fig.suptitle('Optimal Control Problem Solutions', fontsize=16)
    fig.subplots_adjust(wspace=0.3)
    n_fine = max(n_vals)
    solver_fine = FEMSolver(n_fine)
    for alpha in alphas:
        x_int, y_int, u_int = solver_fine.solve_opt_control(alpha, yd)
        u_full = np.zeros_like(solver_fine.nodes)
        u_full[1:-1] = y_int
        axs[0].plot(solver_fine.nodes, u_full, marker='o', linestyle='-', label=f'$\\alpha = {alpha}$')

    x_fine = np.linspace(0, 1, fine_mesh)
    axs[0].plot(x_fine, yd(x_fine), 'k-', linewidth=2, label='Desired $y_d$')
    axs[0].set_title('Optimal Control Solutions')
    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('State $y(x)$')
    axs[0].legend(loc='lower center')
    axs[0].grid(True)
    
    # --- Right Panel: Control solutions ---
    for alpha in alphas:
        x_int, y_int, u_int = solver_fine.solve_opt_control(alpha, yd)
        axs[1].plot(x_int, u_int, marker='o', linestyle='-', label=f'$\\alpha = {alpha}$')
        u_full = np.zeros_like(solver_fine.nodes)
        u_full[1:-1] = u_int
        axs[1].plot(solver_fine.nodes, u_full, marker='o', linestyle='-')
  
    axs[1].set_title('Control Solutions')
    axs[1].set_xlabel('$x$')
    axs[1].set_ylabel('Control $u(x)$')
    axs[1].legend(loc='lower center')
    axs[1].grid(True)
    

    # --- Right Panel: Convergence plot (L2 error vs. mesh size) for each α ---
    for alpha in alphas:
        errors = []
        hs = []
        for n in n_vals:
            solver_temp = FEMSolver(n)
            x_int, y_int, u_int = solver_temp.solve_opt_control(alpha, yd)
            if error_norm == 'H1':
                error = solver_temp.H1_error(y_int, yd(x_int), x_int)
            else:
                error = solver_temp.L2_error(y_int, yd(x_int), x_int)
            errors.append(error)
            hs.append(1.0 / n) 
        hs = np.array(hs)
        errors = np.array(errors)
        p = np.polyfit(np.log(hs), np.log(errors), 1)
        axs[2].loglog(hs, errors, 'o-', label=f'$\\alpha = {alpha}, \\quad \\mathcal{{O}}(h^{{{p[0]:.2f}}})$')
        
    axs[2].set_xlabel('Mesh size $h$')
    axs[2].set_ylabel(f'$\\|e \\|_{{{error_norm}}}$')
    axs[2].set_title('Convergence Plot')
    axs[2].grid(True, which='both', linestyle='--', alpha=0.7)
    axs[2].legend(loc='lower left')
    
    plt.tight_layout()
    if savefig:
        plt.savefig(f'figures/opt_control_plot_{name}.png', dpi=200, bbox_inches='tight')
    plt.show()

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
