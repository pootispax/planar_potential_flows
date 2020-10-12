import numpy as np
from parameters import *


class Matrices:

    def __init__(self):

        self.Nx_quarter = Nx // 4
        self.Ny_quarter = Ny // 4
        self.G = self.build_g()
        self.M = self.build_m(self.G)
        self.cell_coords = self.build_cell_coords(self.G)
        self.phi = self.build_phi()
        self.grad = self.build_gradient() # Own function
        # self.grad = np.gradient(self.phi, 2) # Using numpy gradient function


    def build_g(self):

        G = np.zeros((Nx * h, Ny * h))

        if geometry == 'straight':
            return self.build_g_straight(G).astype(int)

        elif geometry == 'widening':
            return self.build_g_widening(G).astype(int)

        elif geometry == 'shrinkage':
            return self.build_g_shrinkage(G).astype(int)


    # -------------------------------------------------------------------------
    # Build a matrix according to the straight geometry
    def build_g_straight(self, G):

        for j in range(0, Ny * h):
            if j < h:
                for i in range(self.Nx_quarter * h + 1,
                               3 * self.Nx_quarter * h - 1):
                    G[i, j] = 2

            elif j >=(Nx - 1) * h:
                for i in range(self.Nx_quarter * h + 1,
                               3 * self.Nx_quarter * h  - 1):
                    G[i, j] = 3

            else:
                for i in range(self.Nx_quarter * h + 1,
                               3 * self.Nx_quarter * h - 1):
                    G[i, j] = 1

        return G


    # -------------------------------------------------------------------------
    # Build a matrix according to the widening geometry
    def build_g_widening(self, G):

        offset = 0

        for j in range(0, Ny * h):
            if j < h:
                for i in range(Nx * h // 4 + 1, 3 * Nx * h // 4 - 1):
                    G[i, j] = 2

            elif j >= (Ny - 1) * h:
                for i in range(h + 1, (Nx - 1) * h - 1):
                    G[i, j] = 3

            else:
                if j <= self.Ny_quarter * h  - 1:
                    for i in range(self.Nx_quarter * h + 1,
                                   3 * self.Nx_quarter * h - 1):
                        G[i, j] = 1 

                elif j > 2 * self.Ny_quarter * h - 1:
                    for i in range(h + 1, (Nx - 1) * h - 1):
                        G[i, j] = 1

                elif j >= self.Ny_quarter * h and j < 2 * self.Ny_quarter * h:
                    for i in range(self.Nx_quarter * h - offset + 1,
                                   3 * self.Nx_quarter * h + offset - 1):
                        G[i, j] = 1
                    if (j + 1) % h == 0:
                        offset += h 

        return G


    # -------------------------------------------------------------------------
    # Build a matrix according to the shrinkage geometry
    def build_g_shrinkage(self, G):

        offset = 0

        for j in range(0, Ny * h):
            if j < h:
                for i in range(h + 1, (Nx - 1) * h - 1):
                    G[i, j] = 2

            elif j >= (Ny - 1) * h:
                for i in range(self.Nx_quarter * h + 1,
                               3 * self.Nx_quarter * h - 1):
                    G[i, j] = 3

            else:
                if j <= self.Ny_quarter * h - 1:
                    for i in range(h + 1, (Nx - 1) * h - 1):
                        G[i, j] = 1

                elif j > 2 * self.Ny_quarter * h - 1:
                    for i in range(self.Nx_quarter * h + 1,
                                   3 * self.Nx_quarter * h - 1):
                        G[i, j] = 1

                elif j >= self.Ny_quarter * h  and j < 2 * self.Ny_quarter * h:
                    for i in range(h + offset + 1, (Nx - 1) * h - offset - 1):
                        G[i, j] = 1
                    if (j + 1) % h == 0:
                        offset += h 
        
        return G

    # -------------------------------------------------------------------------
    # Builds the matrix M to link each coordinates tp the associated cell
    def build_m(self, G):

        M = np.zeros((Nx * h, Ny * h))
        count = 1

        for j in range(0, Nx * h):
            for i in range(0, Ny * h):
                if self.G[i, j] != 0:
                    M[i, j] = count
                    count += 1

        M = M.astype(int)

        return M


    # -------------------------------------------------------------------------
    # Builds the array_cell list to link each cell to its coordinates
    def build_cell_coords(self, G):

        cell_coords = []
        count = 1

        for j in range(0, Ny * h):
            for i in range(0, Nx * h):
                if self.G[i, j] != 0:
                    cell_coords.append((count, (i, j)))
                    count += 1

        return cell_coords


    # -------------------------------------------------------------------------
    # Builds the matrix A
    def build_a(self):

        A = np.zeros((self.M.max(), self.M.max()))

        for i in range(0, self.M.max()):
            row = self.cell_coords[i][1][0]
            column = self.cell_coords[i][1][1]

            if self.G[row, column] == 1:
                cell_up = self.cell_coords[i][1][0] - 1
                cell_down = self.cell_coords[i][1][0] + 1
                cell_left = self.cell_coords[i][1][1] - 1
                cell_right = self.cell_coords[i][1][1] + 1

                # Checks the walls around the cell
                # Cell above
                if self.G[cell_up, column] != 0 and cell_up >= 0:
                    A[i, i] -= 1
                    A[i, self.M[cell_up, column] - 1] = 1

                # Cell under
                if self.G[cell_down, column] != 0\
                and cell_down <= self.M.max():
                    A[i, i] -= 1
                    A[i, self.M[cell_down, column] - 1] = 1

                # Cell on the left
                if self.G[row, cell_left] != 0 and cell_left >= 0:
                    A[i, i] -= 1
                    A[i, self.M[row, cell_left] - 1] = 1

                # Cell on the right
                if self.G[row, cell_right] != 0\
                and cell_right <= self.M.max():
                    A[i, i] -= 1
                    A[i, self.M[row, cell_right] - 1] = 1

            elif self.G[row, column] == 2:
                A[i, i] = -1
            elif self.G[row, column] == 3:
                A[i, i] = 1

        return A.astype(int)


    # -------------------------------------------------------------------------
    # Builds the vector b
    def build_b(self):

        b = np.zeros((self.M.max(), 1))

        for i in range(self.M.max()):
            if self.G[self.cell_coords[i][1][0],
                      self.cell_coords[i][1][1]] == 2:
                b[i, 0] = inlet

            elif self.G[self.cell_coords[i][1][0],
                        self.cell_coords[i][1][1]] == 3:
                b[i, 0] = outlet

        return b.astype(int)

    # -------------------------------------------------------------------------
    # Build the matrix M
    def build_phi(self):

        A = self.build_a()
        b = self.build_b()
        x = np.linalg.solve(A, b)
        phi = np.zeros((Nx * h, Ny * h))

        for i in range(len(self.cell_coords)):
            row = self.cell_coords[i][1][0]
            column = self.cell_coords[i][1][1]
            phi[row, column] = x[i]

        return phi

    # -------------------------------------------------------------------------
    # Build the gradient of phi
    def build_gradient(self):

        grad_x = np.zeros((Nx * h, Ny * h))
        grad_y = np.zeros((Nx * h, Ny * h))
        grad = []

        for i in range(self.M.max()):
            cell_count_x = 2
            row = self.cell_coords[i][1][0]
            column = self.cell_coords[i][1][1]

            if self.G[row, column] == 1:
                cell_up = self.cell_coords[i][1][0] - 1
                cell_down = self.cell_coords[i][1][0] + 1
                cell_left = self.cell_coords[i][1][1] - 1
                cell_right = self.cell_coords[i][1][1] + 1

                grad_x[row, column] = (self.M[row, cell_right]
                                       - self.M[row, cell_left]) // (2 * h)
                grad_y[row, column] = (self.M[cell_down, column]
                                       - self.M[cell_up, column]) // (2 * h)

            elif self.G[row, column] == 0:
                grad_x[row, column] = 0
                grad_y[row, column] = 0

        grad.append(grad_x)
        grad.append(grad_y)

        return grad

    # def build_numpy_gradient(self):

