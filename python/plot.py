# Class used to plot different things

import numpy as np
import matplotlib.pyplot as plt

import matrices as m
from parameters import Nx, Ny, h, geometry, isopotential_number

np.seterr(divide='ignore', invalid='ignore')


class BuildPlots():

    def plot(self, G, phi, grad, color):

        matrices = m.Matrices()
        fig = plt.figure()

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        ax_list = [ax1, ax2, ax3, ax4]

        for i in ax_list:
            i.xaxis.tick_top()
            i.imshow(matrices.G, cmap='coolwarm')
            self.build_walls(i, G)

        self.plot_contour(ax1, phi, color)
        ax2.quiver(grad[1], grad[0])
        ax3.streamplot(np.linspace(0, Nx * h - 1, Nx * h),
                       np.linspace(0, Ny * h - 1, Ny * h),
                       grad[1], grad[0], linewidth=.75, arrowsize=.75)
        self.plot_contour(ax4, matrices.pressure[0], color)
        ax4.quiver(matrices.pressure[1][0],matrices.pressure[1][1])

        # Saving the figure
        # figname = "../figures/{}_x={}_y={}_h={}.pdf".format(geometry,
        #                                                     Nx, Ny, h)
        figname = "../figures/{}_x={}_y={}.pdf".format(geometry, Nx, Ny)
        plt.savefig(figname)

    # -------------------------------------------------------------------------
    # Builds and plots the borders between walls and fluid cells
    def build_walls(self, ax, G):

        X = np.linspace(0, Nx * h - 1, Nx * h)
        Y = np.linspace(0, Ny * h - 1, Ny * h)
        XX, YY = np.meshgrid(X, Y)

        for j in range(0, Ny * h - 1):
            for i in range(0, Nx * h - 1):
                if G[i, j] == 0:
                    # Builds a wall under the wall cells
                    if G[i + 1, j] != 0:
                        if j == Ny * h - 2:
                            ax.plot(XX[j][j:j + 2] + .5,
                                    YY[i][i:i + 2] + .5,
                                    ls='-', color='black')

                        ax.plot(XX[i][j:j + 2] - .5,
                                YY[i][i:i + 2] + .5,
                                ls='-', color='black')

                    # Builds a wall above the wall cells
                    if G[i - 1, j] != 0:
                        if j == Ny * h - 2:
                            ax.plot(XX[j][j:j + 2] + .5,
                                    YY[i][i:i + 2] - .5,
                                    ls='-', color='black')

                        ax.plot(XX[i][j:j + 2] - .5,
                                YY[i][i:i + 2] - .5,
                                ls='-', color='black')

                    # Builds a wall to the right of the wall cells
                    if G[i, j + 1] != 0:
                        if j == Ny * h - 2:
                            ax.plot(YY[j][j:j + 2] + .5,
                                    XX[i][i:i + 2] + .5,
                                    ls='-', color='black')

                        ax.plot(YY[j][j:j + 2] + .5,
                                XX[i][i:i + 2] - .5,
                                ls='-', color='black')

                    # Builds a wall to the left of the wall cells
                    if G[i, j - 1] != 0:
                        if j == Ny * h - 2:
                            ax.plot(YY[j][j:j + 2] - .5,
                                    XX[i][i:i + 2] + .5,
                                    ls='-', color='black')

                        ax.plot(YY[j][j:j + 2] - .5,
                                XX[i][i:i + 2] - .5,
                                ls='-', color='black')

    # -------------------------------------------------------------------------
    # Plots the contour
    def plot_contour(self, ax, phi, color):

        for j in range(Ny * h):
            for i in range(Nx * h):
                if phi[i, j] == 0:
                    phi[i, j] = np.nan

        ax.contour(phi, levels=isopotential_number,
                   colors=color, linewidths=.75)
