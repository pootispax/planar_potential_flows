import time
import numpy as np
import data_check
from parameters import *


class Matrices:

    def __init__(self):

        self.G = self.build_g(nx, ny, geometry, angle)
        self.M, self.cell_coords = self.build_index_matrices(nx, ny)
        self.b = self.build_b()
        self.A = self.build_a()

    # -------------------------------------------------------------------------
    def make_data(self):

        t_init = time.time()
        print("\nBuilding matrix phi\nSince numpy.linalg.solve() is a slow "
              "function, this can take some time...")

        self.build_phi()
        t_phi = time.time()
        print("Done in {:.3f} seconds.\n\nBuilding gradient..."
              .format(t_phi - t_init))

        self.build_gradient()
        t_grad = time.time()
        print("Done in {:.3f} seconds.\n\nBuilding pressure..."
              .format(t_grad - t_phi))

        self.build_pressure()
        t_pressure = time.time()
        print("Done in {:.3f} seconds.\n\nAll the data has been computed\n"
              .format(t_pressure - t_grad))

    # -------------------------------------------------------------------------
    #  Loads the saved data for later use in the main program
    @staticmethod
    def load_data():

        data = {'G': np.loadtxt('dat/G_{}_{}_{}.dat'
                                .format(geometry, nx, ny),
                                dtype=np.int8),
                'phi': np.loadtxt('dat/phi_{}_{}_{}.dat'
                                  .format(geometry, nx, ny),
                                  dtype=np.float32),
                'grad_x': np.loadtxt('dat/grad_x_{}_{}_{}.dat'
                                     .format(geometry, nx, ny),
                                     dtype=np.float32),
                'grad_y': np.loadtxt('dat/grad_y_{}_{}_{}.dat'
                                     .format(geometry, nx, ny),
                                     dtype=np.float32),
                'grad_norm': np.loadtxt('dat/grad_norm_{}_{}_{}.dat'
                                        .format(geometry, nx, ny),
                                        dtype=np.float32),
                'pressure': np.loadtxt('dat/pressure_{}_{}_{}.dat'
                                       .format(geometry, nx, ny),
                                       dtype=np.float32)}

        return data

    # -------------------------------------------------------------------------
    # Builds the matrix G
    def build_g(self, nx: int, ny: int, geometry: str, angle: np.int8):

        g = np.zeros((ny, nx), dtype=np.int8)

        if geometry == 'straight':
            return self.build_geometry(g, 0)

        elif geometry == 'widening':
            return self.build_geometry(g, np.radians(angle))

        elif geometry == 'shrinkage':
            return self.build_geometry(g, np.radians(-angle))

    # -------------------------------------------------------------------------
    # Builds the matrix G
    @staticmethod
    def build_geometry(g, angle):

        alpha = np.tan(np.abs(angle))

        for i in range(nx):  # Parse the columns of G
            offset = int(i * alpha)
            g[1 + offset: -1 - offset, i] = 1

        # Flip the geometry if angle negative
        if angle > 0:
            g = np.fliplr(g)
        g[:, 0] *= 2  # Inlet
        g[:, -1] *= 3  # Outlet

        if data_check.existing_data():
            np.savetxt('dat/G_{}_{}_{}.dat'
                       .format(geometry, nx, ny), g)

        else:
            pass

        return g

    # Builds the matrix M and the array cell_coords
    def build_index_matrices(self, nx: int, ny: int):

        m = np.zeros(self.G.shape, dtype=np.int)
        count = 0

        for c in range(ny):
            for r in range(nx):
                if self.G[c, r] != 0:
                    m[c, r] = count
                    count += 1
                else:
                    m[c, r] = -1

        cell_coords = np.zeros((m.max(initial=1) + 1, 2), dtype=np.uint)
        count = 0

        for c in range(ny):
            for r in range(nx):
                if m[c, r] != -1:
                    cell_coords[count] = [c, r]
                    count += 1

        return m, cell_coords

    # -------------------------------------------------------------------------
    # Builds the matrix A
    def build_a(self):
        a = np.zeros((self.cell_coords.shape[0],
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
                        a[r, r] -= 1
                        a[r, self.M[n]] = 1
            else:
                a[r, r] = 1

        return a

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
        phi = np.empty(self.G.shape, dtype=np.float32)
        phi.fill(np.nan)

        for i in range(len(x)):
            phi[self.cell_coords[i][0], self.cell_coords[i][1]] = x[i]

        np.savetxt('dat/phi_{}_{}_{}.dat'
                   .format(geometry, nx, ny), phi)

    # -------------------------------------------------------------------------
    # Builds the gradient matrix
    def build_gradient(self):

        phi = np.loadtxt('dat/phi_{}_{}_{}.dat'
                         .format(geometry, nx, ny), dtype=np.float32)

        grad_x = np.empty(self.G.shape, dtype=np.float32)
        grad_x.fill(np.nan)
        grad_y = np.empty(self.G.shape, dtype=np.float32)
        grad_y.fill(np.nan)

        for r in range(self.cell_coords.shape[0]):
            i, j = int(self.cell_coords[r, 0]), int(self.cell_coords[r, 1])

            # Checks for the left and right cells
            if i == 0:
                pass

            elif i == ny - 1:
                grad_y[i, j] = grad_y[i - 1, j]

            if np.isnan(phi[i - 1, j]):
                grad_y[i, j] = (phi[i + 1, j] - phi[i, j]) / (2 * h)

            elif i == ny - 1 or np.isnan(phi[i + 1, j]):
                grad_y[i, j] = (phi[i, j] - phi[i - 1, j]) / (2 * h)

            else:
                grad_y[i, j] = (phi[i + 1, j] - phi[i - 1, j]) / (2 * h)

            # Checks for the up and down cells
            if j == 0:
                grad_x[i, j] = -vx
                grad_y[i, j] = 0

            elif np.isnan(phi[i, j - 1]):
                grad_x[i, j] = (phi[i, j + 1] - phi[i, j]) / (2 * h)

            elif j == nx - 1 or np.isnan(phi[i, j + 1]):
                grad_x[i, j] = (phi[i, j] - phi[i, j - 1]) / (2 * h)

            else:
                grad_x[i, j] = (phi[i, j + 1] - phi[i, j - 1]) / (2 * h)

        grad_norm = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)

        np.savetxt('dat/grad_x_{}_{}_{}.dat'
                   .format(geometry, nx, ny), grad_x)
        np.savetxt('dat/grad_y_{}_{}_{}.dat'
                   .format(geometry, nx, ny), grad_y)
        np.savetxt('dat/grad_norm_{}_{}_{}.dat'
                   .format(geometry, nx, ny), grad_norm)

    # -------------------------------------------------------------------------
    # Builds the pressure field
    def build_pressure(self):

        grad_norm = np.loadtxt('dat/grad_norm_{}_{}_{}.dat'
                               .format(geometry, nx, ny), dtype=np.float32)

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
                   .format(geometry, nx, ny), pressure)
