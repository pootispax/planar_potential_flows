import numpy as np
import matplotlib.pyplot as plt
from parameters import Nx, Ny, geometry


class Plot:

    def plot_graphs(self, display, data):

        fig, ax = plt.subplots()

        ax.imshow(data['G'], cmap='coolwarm', interpolation='none')
        ax.axis('off')

        if display == "potential":
            self.plot_potential(ax, data)
            print('Potential field plotted and saved as {}_{}_Nx={}_Ny={}.pdf'
                  .format(display, geometry, Nx, Ny))

        if display == "velocity":
            self.plot_velocity(ax, data)
            print('Velocity field plotted and saved as {}_{}_Nx={}_Ny={}.pdf'
                  .format(display, geometry, Nx, Ny))

        if display == "streamlines":
            self.plot_streamlines(ax, data)
            print('Streamlines plotted and saved as {}_{}_Nx={}_Ny={}.pdf'
                  .format(display, geometry, Nx, Ny))

        if display == "pressure":
            self.plot_pressure(ax, data)
            im = ax.imshow(data['pressure'], cmap='jet')
            fig.colorbar(im, ax=ax)
            print('Pressure plotted and saved as {}_{}_Nx={}_Ny={}.pdf'
                  .format(display, geometry, Nx, Ny))

        plt.savefig("figures/{}_{}_Nx={}_Ny={}.pdf"
                    .format(display, geometry, Nx, Ny))

    def plot_potential(self, ax, data):

        ax.set_title('Velocity potential field', fontsize=10)
        ax.contour(data['phi'], levels=Nx,
                   colors='green', linewidths=.75)

        return ax

    def plot_velocity(self, ax, data):

        ax.set_title('Velocity field', fontsize=10)

        X = np.linspace(0, Nx - 1, Nx)
        Y = np.linspace(0, Ny - 1, Ny)
        XX, YY = np.meshgrid(X, Y)

        ax.quiver(XX - .25, YY,
                  -data['grad_x']/data['grad_norm'],
                  data['grad_y']/data['grad_norm'])

        return ax

    def plot_streamlines(self, ax, data):

        ax.set_title('Streamlines', fontsize=10)
        ax.streamplot(np.linspace(0, Nx - 1, Nx), np.linspace(0, Ny - 1, Ny),
                      -data['grad_x'], -data['grad_y'],
                      linewidth=.75, arrowsize=.75)

        return ax

    def plot_pressure(self, ax, data):

        ax.set_title('Pressure field and isobaric lines', fontsize=10)

        ax.contour(data['pressure'], levels=10,
                   colors='olivedrab', linewidths=.75)
        return ax
