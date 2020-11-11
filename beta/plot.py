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

        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)
        # step = 100
        # xnew = np.linspace(0, ny - 1, step)
        # ynew = np.linspace(0, nx - 1, step)
        #
        # grad_x = -data['grad_x']
        # grad_y = -data['grad_y']
        #
        # r0 = [1, 1]
        # tmax = 10
        #
        # t = np.linspace(r0, tmax, step**2)
        # grad_x_new = self.interp_nan(x, y, xnew, ynew, grad_x)
        # grad_y_new = self.interp_nan(x, y, xnew, ynew, grad_y)
        #
        # v = np.zeros((len(t), 2), dtype=np.float32)
        # vx = np.reshape(grad_x_new, (len(t), 1))
        # vy = np.reshape(grad_y_new, (len(t), 1))
        # for i in range(len(t)):
        #     v[i, :] = [vx[i], vy[i]]
        #
        # r = self.euler(r0, t, v)
        # print(r)

    @staticmethod
    def plot_pressure(ax, data):

        ax.contour(data['pressure'], levels=10,
                   colors='red', linewidths=.75)

    @staticmethod
    def interp_nan(x, y, xnew, ynew, grad):

        a = grad.copy()
        a_nan = np.zeros_like(a)
        a_nan[np.isnan(a)] = 1
        a[np.isnan(a)] = 0

        interp_a = sp.RectBivariateSpline(y, x, a)
        interp_a_nan = sp.RectBivariateSpline(y, x, a_nan)

        grad_a_new = interp_a(xnew, ynew)
        a_nan_new = interp_a_nan(xnew, ynew)

        grad_a_new[abs(a_nan_new) > 0.5] = np.nan

        return grad_a_new

    def section(self, data, geometry):

        grad_x = data['grad_x'] / data['grad_norm']
        grad_y = data['grad_y'] / data['grad_norm']

        plt.figure()

        if geometry == 'shrinkage' or geometry == 'widening':
            self.section_wid_shrin(grad_x, grad_y)

        if geometry == 'elbow':

            xpart1 = int(nx * 4 / 10)
            ypart1 = int(ny * 4 / 10)
            xpart2 = int(nx * 5 / 10)
            ypart2 = int(ny * 5 / 10)
            xpart3 = int(nx * 6 / 10)
            ypart3 = int(ny * 6 / 10)
            self.section_elbow(grad_x, grad_y, xpart1, ypart1, 1)
            self.section_elbow(grad_x, grad_y, xpart2, ypart3, 2)
            self.section_elbow(grad_x, grad_y, xpart3, ypart3, 3)

    @staticmethod
    def section_wid_shrin(grad_x, grad_y):

        x = np.linspace(0, nx - 1, nx)
        sx = np.zeros(nx)
        sy = np.zeros(nx)
        for i in range(nx - 1):
            sx[i] = -grad_x[int(nx / 2)][i]
            sy[i] = -grad_y[int(nx / 2)][i]

        plt.plot(x, sx)
        plt.plot(x, sy)

        if geometry == 'widening':
            plt.savefig('section_widening.pdf')

        else:
            plt.savefig('section_shrinkage.pdf')

    @staticmethod
    def section_elbow(grad_x, grad_y, xpart, ypart, n):

        plt.figure()

        sx = np.zeros(xpart + ypart)
        sy = np.zeros(xpart + ypart)
        x = np.linspace(0, len(sx) - 1, len(sx))

        for i in range(xpart - 1):
            sx[i] = -grad_x[xpart][i]
            sy[i] = grad_y[xpart][i]

        for i in range(ypart - 1):
            sx[i + xpart - 1] = -grad_x[ypart - i][xpart]
            sy[i + xpart - 1] = grad_y[ypart - i][xpart]

        plt.plot(x, sx)
        plt.plot(x, sy)

        plt.savefig('section_elbow_{}.pdf'.format(n))

    @staticmethod
    def section_obstacle(grad_x, grad_y, xpart, ypart, n):

        x_center = int(nx / 2)
        y_center = int(ny / 2)

        if x_center > y_center:
            r = int(nx * 3 / 10)
        else:
            r = int(ny * 3 / 10)

        sx = np.zeros(nx)
        sy = np.zeros(nx)
        for i in range(nx - 1):
            if not np.isnan(grad_x[xpart][i])\
                and not np.isnan(grad_y[xpart][i]):
                    sx[i] = (-grad_x[xpart][i])
                sy[i] = (-grad_y[xpart][i])



    @staticmethod
    def euler(r0, t, v):

        r = np.zeros((len(t), 2), dtype=np.float32)
        r[0, :] = r0[:]

        for n in range(0, len(t) - 1):
            r[n + 1, :] = r[n, :] + v[n, :] * (t[n + 1] - t[n])

        return r
