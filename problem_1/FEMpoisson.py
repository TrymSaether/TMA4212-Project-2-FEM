import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')


class FEMPoissonSolver:
    def __init__(self, M, f, exact):
        self.M = M
        self.f = f
        self.exact = exact
        self.h = 1.0 / M
        self.N = 2 * M + 1 
        self.nodes = np.linspace(0, 1, self.N) # mesh points
        self.A_ref = np.array([[7/3, -8/3,  1/3],
                               [-8/3, 16/3, -8/3],
                               [ 1/3, -8/3,  7/3]]) # local stiffness matrix
        self.xi_quad = np.array([0.0, 0.5, 1.0]) # ref elements
        self.w_quad  = np.array([1/6, 4/6, 1/6]) # weights for Simpson's rule
        self.A = np.zeros((self.N, self.N))
        self.F = np.zeros(self.N)

    @staticmethod
    def phi(i, xi):
        if i == 0:
            return 2*xi**2 - 3*xi + 1
        elif i == 1:
            return -4 *xi**2 + 4*xi
        elif i == 2:
            return 2 * xi**2 - xi

    def assemble(self):
        for k in range(self.M):
            indices = np.array([2*k, 2*k+1, 2*k+2])
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

    def L2_error(self):
        delta = self.h / 2
        error_sq = 0
        for i in range(self.N):
            err = self.exact(self.nodes[i]) - self.u[i]
            weight = 0.5 if (i == 0 or i == self.N - 1) else 1.0
            error_sq += weight * err**2
        return np.abs(np.sqrt(error_sq * delta))
    
    def plot(self, fine_mesh=200, name='test'):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]}, dpi=200)        
        
        x_fine = np.linspace(0, 1, fine_mesh)
        u_exact = np.array([self.exact(x) for x in x_fine])
        # Plot exact solution and FEM solution
        axs[0].plot(x_fine, u_exact, 'r-', linewidth=2, label='Exact Solution')
        axs[0].plot(self.nodes, self.u, 'bo-', markersize=5, label='FEM Solution')
        axs[0].set_title('FEM vs Exact Solution')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('u(x)')
        axs[0].legend()
        axs[0].grid(True)
        
        M_vals = [10, 20, 40, 80]
        errors = []
        hs = []
        for M in M_vals:
            temp_solver = FEMPoissonSolver(M, self.f, self.exact)
            temp_solver.assemble()
            temp_solver.solve()
            errors.append(temp_solver.L2_error())
            hs.append(1.0/M)
        
        hs = np.array(hs)
        errors = np.array(errors)
        p = np.polyfit(np.log(hs), np.log(errors), 1)
        ref_hs = hs/hs[0]

        # Plot convergence
        axs[1].loglog(hs, errors, 'bo-', linewidth=2, markersize=8, label=f'$\\|e_h\\|_{{L^2}} = \\mathcal{{O}}(h^{{{p[0]:.2f}}})$')
        axs[1].loglog(hs, errors[0]*(ref_hs)**3, 'r--', linewidth=2, label='$\\mathcal{O}(h^3)$')
        axs[1].loglog(hs, errors[0]*(ref_hs)**2, 'g--', linewidth=2, label='$\\mathcal{O}(h^2)$')
        axs[1].set_xlabel('Mesh size $h$')
        axs[1].set_ylabel('$\\|e\\|_{\\mathrm{L}^2}$')
        axs[1].grid(True, which='both', ls='--', alpha=0.7)
        axs[1].legend()
        axs[1].set_title('Convergence Plot')
        
        plt.tight_layout()
        plt.savefig(f'figures/fem_solution_{self.M}_{name}.png')
        plt.show()
        
    def plot_stiffness_matrix_and_load_vector(self, name='test'):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
        
        # Plot stiffness matrix
        ims = ax[0].imshow(self.A, cmap='hot', aspect='auto')
        ax[0].set_title('Stiffness Matrix $\\mathbf{A}$')
        ax[0].set_xlabel('Node Index $j$')
        ax[0].set_ylabel('Node Index $i$')
        cb = plt.colorbar(ims, ax=ax[0], orientation='vertical', pad=0.02)
        cb.set_label('Value')
        ticks = np.linspace(np.min(self.A), np.max(self.A), 5)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{tick:.2f}' for tick in ticks])
        
        # Plot load vector
        bax = ax[1].bar(np.arange(self.N), self.F, color='blue', alpha=0.7)
        ax[1].set_ylim(bottom=0)
        ax[1].set_xlim(left=0, right=self.N)
        ax[1].set_title('Load Vector $\\mathbf{F}$')
        ax[1].set_xlabel('Node Index $i = \\theta(k, j)$')
        ax[1].set_ylabel('Value')
        plt.tight_layout()
        plt.savefig(f'figures/stiffness_matrix_and_load_vector_{self.M}_{name}.png')
        plt.show()
        
