import numpy as np
from scipy import integrate
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
    
    def get_u_exact(self, fine_mesh=200):
        x_fine = np.linspace(0, 1, fine_mesh)
        return self.exact(x_fine)
    
    def L2_error(self):
        u_exact = self.exact(self.nodes)
        l2_error = integrate.simpson((self.u - u_exact)**2, self.nodes)
        return np.sqrt(l2_error)
    
    def H1_error(self):
        h = self.h
        u_exact = self.exact(self.nodes)
        u_exact_grad = np.gradient(u_exact, h)
        u_grad = np.gradient(self.u, h)
        h1_error = integrate.simpson((u_grad - u_exact_grad)**2, self.nodes)
        h1_error += integrate.simpson((self.u - u_exact)**2, self.nodes)
        return np.sqrt(h1_error)
    
    def Hr_error(self, r):
        h = self.h
        u_exact = self.exact(self.nodes)
        error = integrate.simpson((self.u - u_exact)**2, self.nodes)
        
        u_grad = self.u
        u_exact_grad = u_exact
        
        for i in range(r):
            u_grad = np.gradient(u_grad, h)
            u_exact_grad = np.gradient(u_exact_grad, h)
            error += integrate.simpson((u_grad - u_exact_grad)**2, self.nodes)
        return np.sqrt(error)
    
    def plot_solution(self, fine_mesh=200, name='test', savefig=False):
        u_exact = self.get_u_exact(fine_mesh)
        x_fine = np.linspace(0, 1, fine_mesh)
        
        fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

        ax.plot(x_fine, u_exact, 'r-', linewidth=2, label='Exact Solution')
        ax.plot(self.nodes, self.u, 'bo-', markersize=5, label='FEM Solution')
        ax.set_title('FEM vs Exact Solution')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(x)$')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        if savefig:
            plt.savefig(f'figures/fem_plot_{name}_M{self.M}.png', dpi=200)
        plt.show()
        
    def plot(self, fine_mesh=200, name='test', savefig=False):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]}, dpi=150)        
        
        x_fine = np.linspace(0, 1, fine_mesh)
        u_exact = np.array([self.exact(x) for x in x_fine])
        # Plot exact solution and FEM solution
        axs[0].plot(x_fine, u_exact, 'r-', linewidth=2, label='Exact Solution')
        axs[0].plot(self.nodes, self.u, 'bo-', markersize=5, label='FEM Solution')
        axs[0].set_title('FEM vs Exact Solution')
        axs[0].set_xlabel('$x$')
        axs[0].set_ylabel('$u(x)$')
        axs[0].legend()
        axs[0].grid(True)
        
        M_vals = [10, 20, 40, 80, 160]
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
        if savefig:
            plt.savefig(f'figures/fem_plot_convergence_{name}_M{self.M}.png', dpi=200)
        plt.show()
        
    def plot_stiffness_matrix_and_load_vector(self, name='test', savefig=False):
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
        ax[1].set_title('Load Vector $\\mathbf{F}$')
        ax[1].set_xlabel('Node Index $i = \\theta(k, j)$')
        ax[1].set_ylabel('Value')
        plt.tight_layout()
        if savefig:
            plt.savefig(f'figures/stiffness_load_{name}_M{self.M}.png', dpi=200)
        plt.show()
        
    def convergence_test(self):
        M_vals = [2, 4, 8, 16, 32, 64, 128]
        errors_L2 = []
        errors_H1 = []
        hs = []
         # Compute L2 error for each M
        for M in M_vals:
            temp_solver = FEMPoissonSolver(M, self.f, self.exact)
            temp_solver.assemble()
            temp_solver.solve()
        
            errors_L2.append(temp_solver.L2_error())
            errors_H1.append(temp_solver.H1_error())
            hs.append(1.0/M)
        
        hs = np.array(hs)
        errors_L2 = np.array(errors_L2)
        errors_H1 = np.array(errors_H1)
        p_L2 = np.polyfit(np.log(hs), np.log(errors_L2), 1)[0]
        p_H1 = np.polyfit(np.log(hs), np.log(errors_H1), 1)[0]
        return hs, (p_L2, p_H1), (errors_L2, errors_H1)
    
    def print_convergence_table(self):
        hs, (p_L2, p_H1), (errors_L2, errors_H1) = self.convergence_test()
        
        # Calculate width for clean formatting
        width = 80
        col_width = 10
        
        # Print header with centered title
        print("\n" + "Convergence Analysis".center(width))
        print("=" * width)
        headers = ["M", "h", "L2 Error", "Rate L2", "Ratio L2", "H1 Error", "Rate H1", "Ratio H1"]
        print("".join(h.center(col_width) for h in headers))
        print("-" * width)

        # Print rows with aligned columns
        for i in range(len(hs)):
            if i == 0:
                rate_L2 = rate_H1 = ratio_L2 = ratio_H1 = "-"
            else:
                rate_L2 = f"{np.log(errors_L2[i-1]/errors_L2[i])/np.log(2):.2f}"
                rate_H1 = f"{np.log(errors_H1[i-1]/errors_H1[i])/np.log(2):.2f}"
                ratio_L2 = f"{errors_L2[i-1]/errors_L2[i]:.2f}"
                ratio_H1 = f"{errors_H1[i-1]/errors_H1[i]:.2f}"
            
            row = [
                f"{1<<i+1:d}",
                f"{hs[i]:.2e}",
                f"{errors_L2[i]:.2e}",
                rate_L2,
                ratio_L2,
                f"{errors_H1[i]:.2e}",
                rate_H1,
                ratio_H1
            ]
            print("".join(str(item).center(col_width) for item in row))

        # Print summary
        print("=" * width)
        print(f"Overall convergence rates - L2: {p_L2:.2f}, H1: {p_H1:.2f}")
        print(f"Final errors - L2: {errors_L2[-1]:.2e}, H1: {errors_H1[-1]:.2e}")
        print("=" * width + "\n")
        
    def plot_convergence(self, name='test', savefig=False):
        hs, (p_L2, p_H1), (errors_L2, errors_H1) = self.convergence_test()
        ref_hs = hs/hs[0]
        
        # Plot convergence
        fig, ax = plt.subplots(1,2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]}, dpi=200)
        ax[0].loglog(hs, errors_L2, 'bo-', linewidth=2, markersize=8, label=f'$\\|e_h\\|_{{L^2}} = \\mathcal{{O}}(h^{{{p_L2:.2f}}})$')
        ax[0].loglog(hs, errors_H1, 'mo-', linewidth=2, markersize=8, label=f'$\\|e_h\\|_{{H^1}} = \\mathcal{{O}}(h^{{{p_H1:.2f}}})$')
        ax[0].loglog(hs, errors_L2[0]*(ref_hs)**3, 'r--', linewidth=2, label='$\\mathcal{O}(h^3)$', alpha=0.5)
        ax[0].loglog(hs, errors_L2[0]*(ref_hs)**2, 'g--', linewidth=2, label='$\\mathcal{O}(h^2)$', alpha=0.5)

        ax[0].set_xlabel('Mesh size $h$')
        ax[0].set_ylabel('$\\|e\\|_{\\mathrm{L}^2}$ and $\\|e\\|_{\\mathrm{H}^1}$')
        ax[0].grid(True, which='both', ls='--', alpha=0.7)
        ax[0].legend()
        ax[0].set_title('Convergence Plot')
        
        # Plot errors
        ax[1].plot(hs, errors_L2, 'bo-', linewidth=2, markersize=8, label='$\\|e_h\\|_{L^2}$')
        ax[1].plot(hs, errors_H1, 'mo-', linewidth=2, markersize=8, label='$\\|e_h\\|_{H^1}$')
        ax[1].set_xlabel('Mesh size $h$')
        ax[1].set_ylabel('Errors')
        ax[1].grid(True, which='both', ls='--', alpha=0.7)
        ax[1].legend()
        ax[1].set_title('Errors Plot')
        plt.tight_layout()
        if savefig:
            plt.savefig(f'figures/convergence_{name}_M{self.M}.png', dpi=200)
        plt.show()
        
    
    def get_exact_solution(self):
        return np.array([self.exact(x) for x in self.nodes])
    