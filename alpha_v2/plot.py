import numpy as np
import matplotlib.pyplot as plt
from parameters import Nx, Ny, geometry
import matrices as m


class Plot:

    def plot_graphs(self, display):

        matrices = m.Matrices()
        fig, ax = plt.subplots()

        ax.imshow(matrices.G, cmap='coolwarm', interpolation='none')
        ax.axis('off')

        if display == "potential":
            self.plot_potential(ax, "potential")

        if display == "velocity":
            self.plot_velocity(ax, "velocity")

        if display == "streamlines":
            self.plot_streamlines(ax, "streamlines")

        if display == "pressure":
            self.plot_pressure(ax, "pressure")
            im = ax.imshow(matrices.pressure, cmap='jet')
            fig.colorbar(im, ax=ax)

        plt.savefig("figures/{}_{}_Nx={}_Ny={}.pdf"\
                    .format(display, geometry, Nx, Ny))

    def plot_potential(self, ax, display):

        matrices = m.Matrices()

        ax.set_title('Velocity potential field', fontsize=10)
        ax.contour(matrices.phi, levels=Nx,
                   colors='green', linewidths=.75)

        return ax

    def plot_velocity(self, ax, display):

        matrices = m.Matrices()

        ax.set_title('Velocity field', fontsize=10)

        X = np.linspace(0, Nx - 1, Nx)
        Y = np.linspace(0, Ny - 1, Ny)
        XX, YY = np.meshgrid(X, Y)

        ax.quiver(XX - .25, YY, -matrices.grad_own[2], matrices.grad_own[3])

        return ax

    def plot_streamlines(self, ax, display):

        matrices = m.Matrices()

        ax.set_title('Streamlines', fontsize=10)
        ax.streamplot(np.linspace(0, Nx - 1, Nx), np.linspace(0, Ny - 1, Ny),
                      -matrices.grad_own[0], -matrices.grad_own[1],
                      linewidth=.75, arrowsize=.75)

        return ax

    def plot_pressure(self, ax, display):

        matrices = m.Matrices()

        ax.set_title('Pressure field and isobaric lines', fontsize=10)

        ax.contour(matrices.pressure, levels=10,
                   colors='olivedrab', linewidths=.75)
        return ax
