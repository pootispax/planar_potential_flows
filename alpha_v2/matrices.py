import numpy as np
import scipy.linalg as sp
from parameters import *


class Matrices:

    def __init__(self):

        self.G = self.build_g(Nx, Ny, geometry, angle)
        self.M, self.cell_coords = self.build_index_matrices()
        self.b = self.build_b()
        self.A = self.build_a()
        self.phi = self.build_phi()
        self.grad = self.build_gradient()
        self.grad_own = self.build_gradient_own()
        self.pressure = self.build_pressure_field()

    # -------------------------------------------------------------------------
    # Builds the matrix G
    def build_g(self, Nx: int, Ny: int, geometry: str, angle: np.int8):

        G = np.zeros((Ny, Nx), dtype=np.int8)

        if geometry == 'straight':
            return self.build_geometry(G, 0)

        elif geometry == 'widening':
            return self.build_geometry(G, np.radians(angle))

        elif geometry == 'shrinkage':
            return self.build_geometry(G, np.radians(-angle))

    # -------------------------------------------------------------------------
    # Builds the matrix G
    def build_geometry(self, G, angle):

        if abs(angle) > np.arctan((.5 * Ny - 1) / Nx):
            raise ValueError("Valeur d'angle invalide")

        alpha = np.tan(np.abs(angle))

        for i in range(Nx):  # Parse the columns of G
            offset = int(i * alpha)
            G[1 + offset: -1 - offset, i] = 1

        # Flip the geometry if angle negative
        if angle > 0:
            G = np.fliplr(G)
        G[:, 0] *= 2  # Inlet
        G[:, -1] *= 3  # Outlet

        return G

    # Builds the matrix M and the array cell_coords
    def build_index_matrices(self):

        M = np.zeros(self.G.shape, dtype=np.int)
        count = 0

        for c in range(Ny):
            for r in range(Nx):
                if self.G[r, c] != 0:
                    M[r, c] = count
                    count += 1
                else:
                    M[r, c] = -1

        cell_coords = np.zeros((M.max() + 1, 2), dtype=np.uint)
        count = 0

        for c in range(Ny):
            for r in range(Nx):
                if M[r, c] != -1:
                    cell_coords[count] = [r, c]
                    count += 1

        return M, cell_coords

    # -------------------------------------------------------------------------
    # Builds the matrix A
    def build_a(self):
        A = np.zeros((self.cell_coords.shape[0],
                      self.cell_coords.shape[0]), dtype=np.int8)

        for r in range(self.cell_coords.shape[0]):
            i, j = int(self.cell_coords[r, 0]), int(self.cell_coords[r, 1])
            neighbors = []
            if self.G[i, j] != 3:
                if i != 0:
                    neighbors.append((i - 1, j),)

                if i != self.G.shape[0] - 1:
                    neighbors.append((i + 1, j),)

                if j != 0:
                    neighbors.append((i, j - 1),)

                if j != self.G.shape[1] - 1:
                    neighbors.append((i, j + 1),)

                for n in neighbors:
                    if self.G[n]:
                        A[r, r] -= 1
                        A[r, self.M[n]] = 1
            else:
                A[r, r] = 1

        return A

    # -------------------------------------------------------------------------
    # Builds the array b
    def build_b(self):
        b = np.zeros((self.M.max() + 1, 1), dtype=np.float32)

        for i in range(self.M.max() + 1):

            if self.G[self.cell_coords[i][0], self.cell_coords[i][1]] == 2:
                b[i] = -vx * h

            elif self.G[self.cell_coords[i][0], self.cell_coords[i][1]] == 3:
                b[i] = phi_ref

            else:
                b[i] = 0

        return b

    # -------------------------------------------------------------------------
    # Builds the array x
    def build_phi(self):

        x = np.linalg.solve(self.A, self.b)
        phi = np.empty((self.G.shape), dtype=np.float32)
        phi.fill(np.nan)

        for i in range(len(x)):
            phi[self.cell_coords[i][0], self.cell_coords[i][1]] = x[i]
        return phi

    # -------------------------------------------------------------------------
    # Builds the gradient matrix
    def build_gradient(self):
        phi = np.copy(self.phi)

        grad = np.gradient(phi)

        grad_norm = np.sqrt(grad[0]**2 + grad[1]**2)

        return grad, grad / grad_norm

    # -------------------------------------------------------------------------
    # Builds my gradient matrix
    def build_gradient_own(self):

        phi = np.copy(self.phi)

        grad_x = np.empty((self.G.shape), dtype=np.float32)
        grad_x.fill(np.nan)
        grad_y = np.empty((self.G.shape), dtype=np.float32)
        grad_y.fill(np.nan)

        for r in range(self.cell_coords.shape[0]):
            i, j = int(self.cell_coords[r, 0]), int(self.cell_coords[r, 1])

            # Checks for the left and right cells
            if i == 0:
                pass

            elif i == Ny - 1:
                grad_x[i, j] = grad_y[i - 1, j]

            if np.isnan(phi[i - 1, j]):
                grad_y[i, j] = (phi[i + 1, j] - phi[i, j]) / (2 * h)

            elif i == Ny - 1 or np.isnan(phi[i + 1, j]):
                grad_y[i, j] = (phi[i, j] - phi[i - 1, j]) / (2 * h)

            else:
                grad_y[i, j] = (phi[i + 1, j] - phi[i - 1, j]) / (2 * h)

            # Checks for the up and down cells
            if j == 0 or np.isnan(phi[i, j - 1]):
                grad_x[i, j] = (phi[i, j + 1] - phi[i, j]) / (2 * h)

            elif j == Nx - 1 or np.isnan(phi[i, j + 1]):
                grad_x[i, j] = (phi[i, j] - phi[i, j - 1]) / (2 * h)

            else:
                grad_x[i, j] = (phi[i, j + 1] - phi[i, j - 1]) / (2 * h)

        grad_norm = np.sqrt(grad_x**2 + grad_y**2)

        return grad_x, grad_y,\
            grad_x / grad_norm, grad_y / grad_norm, grad_norm

    # -------------------------------------------------------------------------
    # Builds the pressure field
    def build_pressure_field(self):

        rho = 1
        pressure_init = 1
        pressure = np.empty(self.G.shape, dtype=np.float32)
        pressure.fill(np.nan)

        # pressure_cst = pressure_init + rho\
        #     * self.grad_own[4][self.cell_coords[0, 0],
        #                        self.cell_coords[0, 1]]**2 / 2

        pressure_cst = pressure_init + rho * vx**2 / 2

        for i in range(self.cell_coords.shape[0]):
            r = self.cell_coords[i][0]
            c = self.cell_coords[i][1]

            if self.G[r, c] == 2:
                pressure[r, c] = pressure_init

            else:
                pressure[r, c] = pressure_cst - rho\
                    * self.grad_own[4][r, c]**2 / 2

        return pressure
