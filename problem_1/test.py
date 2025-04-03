import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")

class FEMSolver:
    def __init__(self, nodes=None, M=None, f=None, exact=None, du_exact=None, ddu_exact=None, domain=(0, 1)):
        """
        Unified FEM solver for Poisson and OCP using quadratic elements.
        
        Args:
            nodes: Custom node positions (for non-uniform mesh)
            M: Number of elements (for uniform mesh when nodes=None)
            f: Source function (optional)
            exact: Exact solution function (optional)
            du_exact: First derivative of exact solution (optional)
            ddu_exact: Second derivative of exact solution (optional)
            domain: Domain boundaries (default: (0,1))
        """
        # Store functions
        self.f = f
        self.exact = exact
        self.du_exact = du_exact
        self.ddu_exact = ddu_exact
        self.domain = domain
        
        # Create mesh (uniform or non-uniform)
        if nodes is not None:
            # Use provided nodes (non-uniform mesh)
            self.nodes = np.array(nodes)
            # Verify node count is compatible with quadratic elements (must be odd)
            if len(self.nodes) % 2 == 0:
                raise ValueError("For quadratic elements, number of nodes must be odd (2M+1)")
            self.N = len(self.nodes)
            self.M = (self.N - 1) // 2 
        else:
            # Create uniform mesh
            self.M = M
            self.N = 2 * M + 1 
            self.nodes = np.linspace(domain[0], domain[1], self.N)
        
        # Create elements (triplets of nodes)
        self.elements = [(2*i, 2*i+1, 2*i+2) for i in range(self.M)]
        self.ndof = self.N
        
        # Simpson's rule quadrature on reference element [0,1]
        self.xi_quad = np.array([0.0, 0.5, 1.0])
        self.w_quad = np.array([1/6, 4/6, 1/6])
        
        # Initialize matrices and vectors
        self.A = None
        self.b = None
        self.u = np.zeros(self.N)
        
        # Interior nodes for boundary conditions
        self.N_int = len(self.nodes) - 2
        self.int_idx = slice(1, -1)
        self.free = np.arange(self.N)[self.int_idx] 
    
    @staticmethod
    def basis_func(order=0):
        """Returns the basis function and its derivatives up to given order."""
        if order == 0:
            return [
                lambda xi: 2.0 * xi**2 - 3.0 * xi + 1.0,
                lambda xi: -4.0 * xi**2 + 4.0 * xi,
                lambda xi: 2.0 * xi**2 - xi
            ]
        elif order == 1:
            return [
                lambda xi: 4.0 * xi - 3.0,
                lambda xi: -8.0 * xi + 4.0,
                lambda xi: 4.0 * xi - 1.0
            ]
        elif order == 2:
            return [
                lambda xi: 4.0,
                lambda xi: -8.0,
                lambda xi: 4.0
            ]
        else:
            return [lambda xi: 0.0] * 3
    
    def assemble_matrix(self, matrix_type="stiffness"):
        use_stiffness, use_mass, use_load = [matrix_type.lower() == t for t in ("stiffness", "mass", "load")]
        M = np.zeros((self.N, self.N)) if use_stiffness or use_mass else np.zeros(self.N)
        
        basis = self.basis_func(0)
        dbasis = self.basis_func(1)
        
        for el in self.elements:
            xL, xM, xR = self.nodes[el[0]], self.nodes[el[1]], self.nodes[el[2]]
            h_k = xR - xL
            if abs(xM - (xL + h_k/2)) > 1e-10:
                print(f"Warning: Middle node not at midpoint for element {el}. Using approximation.")
            
            locM = np.zeros((3, 3)) if use_stiffness or use_mass else np.zeros(3)
            for i in range(3):
                for j in range(3):
                    for q, xi in enumerate(self.xi_quad):
                        w = self.w_quad[q] * h_k
                        if use_mass:  # Mass matrix
                            locM[i, j] += basis[i](xi) * basis[j](xi) * w
                        elif use_stiffness:
                            locM[i, j] += dbasis[i](xi) * dbasis[j](xi) * w / h_k
                        elif use_load:  # Load vector
                            locM[i] += basis[i](xi) * self.f(xL + h_k * xi) * w
            
            for i_loc, i_glob in enumerate(el):
                if use_load:
                    M[i_glob] += locM[i_loc]
                else:
                    for j_loc, j_glob in enumerate(el):
                        M[i_glob, j_glob] += locM[i_loc, j_loc]
        return M if use_load else csr_matrix(M)
    
    def stiffness_matrix(self):
        return self.assemble_matrix("stiffness")[self.free][:, self.free]
    
    def mass_matrix(self):
        return self.assemble_matrix("mass")[self.free][:, self.free]
    def load_vector(self):
        return self.assemble_matrix("load")[self.free]
        
    def solve_poisson(self, f_func=None):
        """
        Solve the Poisson equation -Δu = f.
        
        Args:
            f_func: Source function (optional)
            
        Returns:
            x, u where x are all nodes and u is the full solution
        """
        if f_func is not None:
            self.f = f_func
            
        # Assemble system
        self.A = self.stiffness_matrix()
        self.b = self.load_vector()
        
        # Solve system
        u_int = spsolve(self.A, self.b)
        
        # Reconstruct full solution
        self.u[self.free] = u_int
        
        return self.nodes, self.u
    
    def solve_opt_control(self, alpha, yd):
        """
        Solve the optimal control problem:
          min_{y,u in V_h}  (1/2)||y - ȳ_d||^2 + (α/2)||u||^2
          s.t. a(y, v) = (u, v) for all v in V_h.
        Here ȳ_d is the interpolation of y_d onto V_h.
        Returns: (x, y_h, u_h) with x the interior nodes.
        """
        
        B = self.stiffness_matrix() # B: stiffness matrix from Poisson (using derivatives)
        F = self.mass_matrix()  # F: mass matrix for L2 inner product
        x = self.nodes[self.free]
        # Desired state at interior nodes
        Yd = yd(x)
        
        # Build block system:
        #   [   M       αK   ] [ y ] = [ M*Yd ]
        #   [   K       -M   ] [ u ]   [   0  ]
        
        # Build block system
        A = csr_matrix(np.block([[F, alpha * B], [B, -F]]))
        b = np.concatenate([F @ Yd, np.zeros_like(Yd)])
        # Solve system
        sol = spsolve(A, b)
        (y_h, u_h) = np.split(sol, 2)
        return x, y_h, u_h
                 
    def error(self):
        """Compute L2 and H1 errors of the FEM solution."""
        if self.u is None:
            raise ValueError("No solution available. Call solve_poisson first.")
            
        if self.exact is None or self.du_exact is None:
            raise ValueError("Exact solution and derivative required for error calculation.")
            
        L2_error_sq = 0.0
        H1_error_sq = 0.0
        
        basis = self.basis_func(0)
        dbasis = self.basis_func(1)
        
        # Loop over each element
        for k in range(self.M):
            idx = np.array([2*k, 2*k+1, 2*k+2])
            xL = self.nodes[idx[0]]
            xR = self.nodes[idx[2]]
            h_k = xR - xL  # Element-specific length
            
            for q, xi in enumerate(self.xi_quad):
                w = self.w_quad[q] * h_k
                x = xL + h_k * xi
                
                u_h = 0.0
                du_h = 0.0
                for i in range(3):
                    u_h += self.u[idx[i]] * basis[i](xi)
                    du_h += self.u[idx[i]] * dbasis[i](xi) / h_k
                
                # Compute exact solution and derivative
                u_ex = self.exact(x)
                du_ex = self.du_exact(x)
                
                # Accumulate errors with correct scaling
                L2_error_sq += w * (u_ex - u_h)**2
                H1_error_sq += w * (du_ex - du_h)**2
        
        # Take square roots for final norms
        L2_error = np.sqrt(L2_error_sq)
        H1_error = np.sqrt(H1_error_sq)
        
        return L2_error, H1_error

def create_nonuniform_mesh(M, domain=(0,1), refinement=2.0):
    # First create endpoint nodes with desired spacing
    t = np.linspace(0, 1, M+1)
    endpoints = domain[0] + (domain[1] - domain[0]) * (t**refinement)
    
    # Create full node array
    nodes = np.zeros(2*M + 1)
    
    # Set element endpoints
    nodes[::2] = endpoints
    
    # Place middle nodes exactly at midpoints
    nodes[1::2] = (nodes[::2][1:] + nodes[::2][:-1]) / 2
    
    return nodes


# Example usage
if __name__ == "__main__":
    # Define parameters
    M = 4
    alpha = 1.0
    domain = (0, 1)
    
    # Create non-uniform mesh
    nodes = create_nonuniform_mesh(M, domain=domain)
    
    # Define source function and exact solution
    f_func = lambda x: -np.pi**2 * np.sin(np.pi * x)
    exact_solution = lambda x: np.sin(np.pi * x)
    du_exact_solution = lambda x: np.pi * np.cos(np.pi * x)
    
    # Create FEM solver instance
    fem_solver = FEMSolver(nodes=nodes, f=f_func, exact=exact_solution, du_exact=du_exact_solution, domain=domain)
    
    # Solve Poisson equation
    x, u_h = fem_solver.solve_poisson()
    
    # Compute errors
    L2_error, H1_error = fem_solver.error()
    
    print(f"L2 Error: {L2_error:.6f}, H1 Error: {H1_error:.6f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(x, u_h, label="FEM Solution", marker='o')
    plt.plot(x, exact_solution(x), label="Exact Solution", linestyle='--')
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("FEM Solution vs Exact Solution")
    plt.legend()
    plt.grid()
    plt.show()