# Class used to plot different things

import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from matrices import *

class BuildPlots():


    def plot(self, G, phi, color):

        matrices = Matrices()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.tick_top()

        ax.imshow(matrices.G, cmap='coolwarm')
        self.build_walls(ax, G)
        self.plot_contour(phi, color)

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
    def plot_contour(self, phi, color):
        X = np.linspace(0, Nx * h - 1, Nx * h)
        Y = np.linspace(0, Ny * h - 1, Ny * h)
        plt.contour(X, Y, phi, colors=color, linewidths=1)
