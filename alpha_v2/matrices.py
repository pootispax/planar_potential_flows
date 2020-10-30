import numpy as np
from parameters import *


class Matrices:

    def __init__(self):

        self.G = self.build_g(Nx, Ny, geometry, angle)
        self.M, self.cell_coords = self.build_index_matrices(Nx, Ny)
        self.b = self.build_b()
        self.A = self.build_a()

    # -------------------------------------------------------------------------
    def make_data(self):

        print("Building matrix phi\nSince numpy.linalg.solve() is a slow "
              "function, this can take some time...")
        self.build_phi()
        print("Done\nBuilding gradient...")
        self.build_gradient()
        print("Done\nBuilding pressure...")
        self.build_pressure()
        print("Done\nAll the data has been computed")

    # -------------------------------------------------------------------------
    #  Loads the saved data for later use in the main program
    def load_data(self):

        data = {'G': np.loadtxt('dat/G_{}_{}_{}.dat'
                                .format(geometry, Nx, Ny),
                                dtype=np.int8),
                'phi': np.loadtxt('dat/phi_{}_{}_{}.dat'
                                  .format(geometry, Nx, Ny),
                                  dtype=np.float32),
                'grad_x': np.loadtxt('dat/grad_x_{}_{}_{}.dat'
                                     .format(geometry, Nx, Ny),
                                     dtype=np.float32),
                'grad_y': np.loadtxt('dat/grad_y_{}_{}_{}.dat'
                                     .format(geometry, Nx, Ny),
                                     dtype=np.float32),
                'grad_norm': np.loadtxt('dat/grad_norm_{}_{}_{}.dat'
                                        .format(geometry, Nx, Ny),
                                        dtype=np.float32),
                'pressure': np.loadtxt('dat/pressure_{}_{}_{}.dat'
                                       .format(geometry, Nx, Ny),
                                       dtype=np.float32)}

        return data

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

        alpha = np.tan(np.abs(angle))

        for i in range(Nx):  # Parse the columns of G
            offset = int(i * alpha)
            G[1 + offset: -1 - offset, i] = 1

        # Flip the geometry if angle negative
        if angle > 0:
            G = np.fliplr(G)
        G[:, 0] *= 2  # Inlet
        G[:, -1] *= 3  # Outlet

        if recompute:
            np.savetxt('dat/G_{}_{}_{}.dat'
                       .format(geometry, Nx, Ny), G)

        else:
            pass

        return G

    # Builds the matrix M and the array cell_coords
    def build_index_matrices(self, Nx: int, Ny: int):

        M = np.zeros(self.G.shape, dtype=np.int)
        count = 0

        for c in range(Ny):
            for r in range(Nx):
                if self.G[c, r] != 0:
                    M[c, r] = count
                    count += 1
                else:
                    M[c, r] = -1

        cell_coords = np.zeros((M.max() + 1, 2), dtype=np.uint)
        count = 0

        for c in range(Ny):
            for r in range(Nx):
                if M[c, r] != -1:
                    cell_coords[count] = [c, r]
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
        b = np.zeros((self.cell_coords.shape[0], 1), dtype=np.float32)

        for i in range(self.cell_coords.shape[0]):

            if self.G[self.cell_coords[i][0], self.cell_coords[i][1]] == 2:
                b[i] = -vx * h

            elif self.G[self.cell_coords[i][0], self.cell_coords[i][1]] == 3:
                b[i] = phi_ref

            else:
                pass

        return b

    # -------------------------------------------------------------------------
    # Builds the array x
    def build_phi(self):

        x = np.linalg.solve(self.A, self.b)
        phi = np.empty((self.G.shape), dtype=np.float32)
        phi.fill(np.nan)

        for i in range(len(x)):
            phi[self.cell_coords[i][0], self.cell_coords[i][1]] = x[i]

        np.savetxt('dat/phi_{}_{}_{}.dat'
                   .format(geometry, Nx, Ny), phi)

    # -------------------------------------------------------------------------
    # Builds the gradient matrix
    def build_gradient(self):

        phi = np.loadtxt('dat/phi_{}_{}_{}.dat'
                         .format(geometry, Nx, Ny), dtype=np.float32)

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
                grad_y[i, j] = grad_y[i - 1, j]

            if np.isnan(phi[i - 1, j]):
                grad_y[i, j] = (phi[i + 1, j] - phi[i, j]) / (2 * h)

            elif i == Ny - 1 or np.isnan(phi[i + 1, j]):
                grad_y[i, j] = (phi[i, j] - phi[i - 1, j]) / (2 * h)

            else:
                grad_y[i, j] = (phi[i + 1, j] - phi[i - 1, j]) / (2 * h)

            # Checks for the up and down cells
            if j == 0:
                grad_x[i, j] = -vx
                grad_y[i, j] = 0

            elif np.isnan(phi[i, j - 1]):
                grad_x[i, j] = (phi[i, j + 1] - phi[i, j]) / (2 * h)

            elif j == Nx - 1 or np.isnan(phi[i, j + 1]):
                grad_x[i, j] = (phi[i, j] - phi[i, j - 1]) / (2 * h)

            else:
                grad_x[i, j] = (phi[i, j + 1] - phi[i, j - 1]) / (2 * h)

        grad_norm = np.sqrt(grad_x**2 + grad_y**2)

        np.savetxt('dat/grad_x_{}_{}_{}.dat'
                   .format(geometry, Nx, Ny), grad_x)
        np.savetxt('dat/grad_y_{}_{}_{}.dat'
                   .format(geometry, Nx, Ny), grad_y)
        np.savetxt('dat/grad_norm_{}_{}_{}.dat'
                   .format(geometry, Nx, Ny), grad_norm)

    # -------------------------------------------------------------------------
    # Builds the pressure field
    def build_pressure(self):

        grad_norm = np.loadtxt('dat/grad_norm_{}_{}_{}.dat'
                               .format(geometry, Nx, Ny), dtype=np.float32)

        rho = 1
        pressure_init = 1
        pressure = np.empty(self.G.shape, dtype=np.float32)
        pressure.fill(np.nan)

        pressure_cst = pressure_init - rho * vx**2 / 2

        for r in range(self.cell_coords.shape[0]):
            i, j = int(self.cell_coords[r, 0]), int(self.cell_coords[r, 1])

            if self.G[i, j] == 2:
                pressure[i, j] = pressure_init

            else:
                pressure[i, j] = pressure_cst + rho\
                    * grad_norm[i, j]**2 / 2

        np.savetxt('dat/pressure_{}_{}_{}.dat'
                   .format(geometry, Nx, Ny), pressure)
