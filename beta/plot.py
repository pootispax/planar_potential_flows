import numpy as np
import scipy.interpolate as sp
import matplotlib.pyplot as plt
from parameters import nx, ny, geometry, h


class Plot:

    def plot_graphs(self, display, data, interp=0):

        interp_bool = False
        fig, ax = plt.subplots()

        ax.imshow(data['G'], cmap='coolwarm', interpolation='none')
        ax.axis('off')

        if display == "potential":
            self.plot_potential(ax, data)
            im = ax.imshow(data['phi'], cmap='jet')
            fig.colorbar(im, ax=ax)
            print('Potential field plotted and saved as {}_{}_Nx={}_Ny={}.pdf'
                  .format(display, geometry, nx, ny))

        if display == "velocity":
            interp = self.plot_velocity(ax, data)
            im = ax.imshow(data['grad_norm'], cmap='jet')
            fig.colorbar(im, ax=ax)
            print('Velocity field plotted and saved as {}_{}_Nx={}_Ny={}.pdf'
                  .format(display, geometry, nx, ny))
            interp_bool = True

        if display == "streamlines":
            self.plot_streamlines(ax, data, interp)
            print('Streamlines plotted and saved as {}_{}_Nx={}_Ny={}.pdf'
                  .format(display, geometry, nx, ny))

        if display == "pressure":
            self.plot_pressure(ax, data)
            im = ax.imshow(data['pressure'], cmap='jet')
            fig.colorbar(im, ax=ax)
            print('Pressure plotted and saved as {}_{}_Nx={}_Ny={}.pdf'
                  .format(display, geometry, nx, ny))

        plt.savefig("figures/{}_{}_Nx={}_Ny={}.pdf"
                    .format(display, geometry, nx, ny))

        if interp_bool:
            return interp

    @staticmethod
    def plot_potential(ax, data):

        ax.imshow(data['phi'], cmap='jet')

        ax.contour(data['phi'], levels=nx,
                   colors='green', linewidths=.75)

    @staticmethod
    def plot_velocity(ax, data):

        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)
        grad_x = -data['grad_x']
        grad_y = data['grad_y']
        grad_norm = data['grad_norm']
        x_nan = np.zeros_like(grad_x)
        y_nan = np.zeros_like(grad_y)
        norm_nan = np.zeros_like(grad_norm)

        x_nan[np.isnan(grad_x)] = 1
        y_nan[np.isnan(grad_y)] = 1
        norm_nan[np.isnan(grad_norm)] = 1

        grad_x[np.isnan(grad_x)] = 0
        grad_y[np.isnan(grad_y)] = 0
        grad_norm[np.isnan(grad_norm)] = 0

        interp_x = sp.RectBivariateSpline(y, x, grad_x)
        interp_y = sp.RectBivariateSpline(y, x, grad_y)
        interp_norm = sp.RectBivariateSpline(y, x, grad_norm)

        interp_x_nan = sp.RectBivariateSpline(y, x, x_nan)
        interp_y_nan = sp.RectBivariateSpline(y, x, y_nan)
        interp_norm_nan = sp.RectBivariateSpline(y, x, norm_nan)

        xnew = np.linspace(0, ny - 1, ny // h)
        ynew = np.linspace(0, nx - 1, nx // h)
        grad_x_new = interp_x(xnew, ynew)
        grad_y_new = interp_y(xnew, ynew)
        grad_norm_new = interp_norm(xnew, ynew)
        x_nan_new = interp_x_nan(xnew, ynew)
        y_nan_new = interp_y_nan(xnew, ynew)
        norm_nan_new = interp_norm_nan(xnew, ynew)

        grad_x_new[x_nan_new > 0.5] = np.nan
        grad_y_new[y_nan_new > 0.5] = np.nan
        grad_norm_new[norm_nan_new > 0.5] = np.nan

        xx, yy = np.meshgrid(ynew, xnew)
        ax.quiver(xx, yy,
                  grad_x_new / grad_norm_new, grad_y_new / grad_norm_new)

        interp = {'xx': xx,
                  'yy': yy,
                  'grad_x_new': grad_x_new,
                  'grad_y_new': grad_y_new,
                  'grad_norm_new': grad_norm_new}

        return interp

    @staticmethod
    def plot_streamlines(ax, data, interp):

        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)
        grad_x = -data['grad_x']
        grad_y = -data['grad_y']

        # hx = interp['xx'].shape()
        # hy = interp['yy'].shape()

    @staticmethod
    def plot_pressure(ax, data):

        ax.contour(data['pressure'], levels=10,
                   colors='red', linewidths=.75)
