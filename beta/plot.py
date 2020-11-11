import numpy as np
import scipy.interpolate as sp
import matplotlib.pyplot as plt
from parameters import nx, ny, geometry, h


class Plot:

    def plot_graphs(self, display, data, interp=0):

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
            self.plot_velocity(ax, data)
            im = ax.imshow(data['grad_norm'], cmap='jet')
            fig.colorbar(im, ax=ax)
            print('Velocity field plotted and saved as {}_{}_Nx={}_Ny={}.pdf'
                  .format(display, geometry, nx, ny))

        if display == "streamlines":
            self.plot_streamlines(ax, data)
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

    @staticmethod
    def plot_potential(ax, data):

        ax.imshow(data['phi'], cmap='jet')

        ax.contour(data['phi'], levels=nx,
                   colors='green', linewidths=.75)

    def plot_velocity(self, ax, data):

        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)
        xnew = np.linspace(0, ny - 1, ny // h)
        ynew = np.linspace(0, nx - 1, nx // h)

        grad_x = -data['grad_x']
        grad_y = data['grad_y']
        grad_norm = data['grad_norm']

        grad_x_new = self.interp_nan(x, y, xnew, ynew, grad_x)
        grad_y_new = self.interp_nan(x, y, xnew, ynew, grad_y)
        grad_norm_new = self.interp_nan(x, y, xnew, ynew, grad_norm)

        xx, yy = np.meshgrid(ynew, xnew)
        ax.quiver(xx, yy,
                  grad_x_new / grad_norm_new, grad_y_new / grad_norm_new)

    def plot_streamlines(self, ax, data):

        grad_x = -data['grad_x']
        grad_y = -data['grad_y']

        hx = 1
        hy = 1

        t0 = 0
        tmax = 10
        step = 1000
        t = np.linspace(0, tmax, step)
        y = self.euler(t0, 10, 1000, f)
        print('stop')

    @staticmethod
    def plot_pressure(ax, data):

        ax.contour(data['pressure'], levels=10,
                   colors='red', linewidths=.75)

    @staticmethod
    def interp_nan(x, y, xnew, ynew, a):

        a_nan = np.zeros_like(a)
        a_nan[np.isnan(a)] = 1
        a[np.isnan(a)] = 0

        interp_a = sp.RectBivariateSpline(y, x, a)
        interp_a_nan = sp.RectBivariateSpline(y, x, a_nan)

        grad_a_new = interp_a(xnew, ynew)
        a_nan_new = interp_a_nan(xnew, ynew)

        grad_a_new[a_nan_new > 0.5] = np.nan

        return grad_a_new

    @staticmethod
    def euler(r0, t, f):

        r = np.zeros(len(t), 2)
        r[0] = r0

        for n in range(0, len(t) - 1):
            r[n + 1] = r[n] + f(r[n], t[n]) * (t[n + 1] - t[n])

        return r
