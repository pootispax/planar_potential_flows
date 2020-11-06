import numpy as np
import scipy.interpolate as sp
import matplotlib.pyplot as plt
from parameters import Nx, Ny, geometry, h


class Plot:

    def plot_graphs(self, display, data):

        fig, ax = plt.subplots()

        ax.imshow(data['G'], cmap='coolwarm', interpolation='none')
        ax.axis('off')

        if display == "potential":
            self.plot_potential(ax, data)
            im = ax.imshow(data['phi'], cmap='jet')
            fig.colorbar(im, ax=ax)
            print('Potential field plotted and saved as {}_{}_Nx={}_Ny={}.pdf'
                  .format(display, geometry, Nx, Ny))

        if display == "velocity":
            self.plot_velocity(ax, data)
            im = ax.imshow(data['grad_norm'], cmap='jet')
            fig.colorbar(im, ax=ax)
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

    @staticmethod
    def plot_potential(ax, data):

        # ax.set_title('Velocity potential field', fontsize=10)
        ax.imshow(data['phi'], cmap='jet')

        ax.contour(data['phi'], levels=Nx,
                   colors='green', linewidths=.75)

        return ax

    @staticmethod
    def plot_velocity(ax, data):

        x = np.linspace(0, Nx - 1, Nx)
        y = np.linspace(0, Ny - 1, Ny)
        grad_x = np.nan_to_num(-data['grad_x']/data['grad_norm'], nan=0)
        grad_y = np.nan_to_num(data['grad_y']/data['grad_norm'], nan=0)

        interp_x = sp.RectBivariateSpline(y, x, grad_x)
        interp_y = sp.RectBivariateSpline(y, x, grad_y)

        xnew = np.linspace(0, Nx - 1, Nx // h)
        ynew = np.linspace(0, Ny - 1, Ny // h)
        grad_x_new = interp_x(xnew, ynew)
        grad_y_new = interp_y(xnew, ynew)

        for j in range(Ny // h):
            for i in range(Nx // h):
                if grad_x_new[i, j] == 0:
                    grad_x_new[i, j] = np.nan

                if grad_y_new[i, j] == 0:
                    grad_y_new[i, j] = np.nan

        xx, yy = np.meshgrid(xnew, ynew)
        ax.quiver(xx, yy, grad_x_new, grad_y_new)

        return ax

    @staticmethod
    def plot_streamlines(ax, data):

        # ax.set_title('Streamlines', fontsize=10)
        ax.streamplot(np.linspace(0, Nx - 1, Nx), np.linspace(0, Ny - 1, Ny),
                      -data['grad_x'], -data['grad_y'],
                      linewidth=.75, arrowsize=.75)

        return ax

    @staticmethod
    def plot_pressure(ax, data):

        # ax.set_title('Pressure field and isobaric lines', fontsize=10)

        ax.contour(data['pressure'], levels=10,
                   colors='red', linewidths=.75)
        return ax
