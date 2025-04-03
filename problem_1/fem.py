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
        # print(f"elements: {elements}, shapes: (U.shape = {U.shape}, x.shape = {x.shape}, uh.shape = {uh.shape})")
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
