# Class used to plot different things

import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from matrices import *

class BuildPlots():


    def plot(self, G, phi, grad, color):

        matrices = Matrices()
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.xaxis.tick_top()
        ax2.xaxis.tick_top()

        ax1.imshow(matrices.G, cmap='coolwarm')
        ax2.imshow(matrices.G, cmap='coolwarm')
        self.build_walls(ax1, G)
        self.build_walls(ax2, G)
        self.plot_contour(ax1, phi, color)

        # Normalization of the vectors
        r = (grad[0]**2 + grad[1]**2)**.5
        
        ax2.quiver(grad[0] / r, grad[1] / r)


        # Saving the figure
        figname = "../figures/{}_x={}_y={}_h={}.pdf".format(geometry,
                                                            Nx, Ny, h)
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
                                XX[i][i:i + 2] - .5 ,
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
        X = np.linspace(0, Nx * h - 1, Nx * h)
        Y = np.linspace(0, Ny * h - 1, Ny * h)
        ax.contour(X, Y, phi, colors=color, linewidths=1)
