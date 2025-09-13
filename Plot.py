import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

class Plot:
    """Class for plotting results of the CFD simulation.
    """
    
    def __init__(self):
        pass

    def plot_evolution_during_loop(self, V_for_all_ite, rho_for_all_ite, T_for_all_ite):
        """Plot the evolution of velocity, density, temperature and mach during iterations.
        Parameters
        ----------
        V_for_all_ite : np.ndarray
            Array of velocity profiles at each iteration.
        rho_for_all_ite : np.ndarray
            Array of density profiles at each iteration.
        T_for_all_ite : np.ndarray
            Array of temperature profiles at each iteration.
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Evolution During Loop', fontsize=16)

        # plot velocity evolution at throat
        axs[0, 0].plot(V_for_all_ite[:, len(V_for_all_ite[0])//2])
        axs[0, 0].grid()
        axs[0, 0].set_title('Velocity Evolution')
        axs[0, 0].set_xlabel('Iteration')
        axs[0, 0].set_ylabel('$V/V_0$ [-]')

        # plot density evolution at throat
        axs[0, 1].plot(rho_for_all_ite[:, len(rho_for_all_ite[0])//2])
        axs[0, 1].grid()
        axs[0, 1].set_title('Density Evolution')
        axs[0, 1].set_xlabel('Iteration')
        axs[0, 1].set_ylabel(r'${\rho}/{\rho_0}$ [-]')

        # plot temperature evolution at throat
        axs[1, 0].plot(T_for_all_ite[:, len(T_for_all_ite[0])//2])
        axs[1, 0].grid()
        axs[1, 0].set_title('Temperature Evolution')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('$T/T_0$ [-]')

        # plot Mach evolution at throat
        M = V_for_all_ite[:, len(V_for_all_ite[0])//2] / (T_for_all_ite[:, len(T_for_all_ite[0])//2]**(1/2))
        axs[1, 1].plot(M)
        axs[1, 1].grid()
        axs[1, 1].set_title('Mach Evolution')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('$Mach$ [-]')

    def plot_final_state(self, V, rho, T):
        """Plot the final state profiles of velocity, density, and temperature.
        Parameters
        ----------
        V : np.ndarray
            Final velocity profile.
        rho : np.ndarray
            Final density profile.
        T : np.ndarray
            Final temperature profile.
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Final State Profiles', fontsize=16)

        # plot velocity profile
        axs[0, 0].plot(V)
        axs[0, 0].grid()
        axs[0, 0].set_title('Velocity Profile')
        axs[0, 0].set_xlabel('Position')
        axs[0, 0].set_ylabel('$V/V_0$ [-]')

        # plot density profile
        axs[0, 1].plot(rho)
        axs[0, 1].grid()
        axs[0, 1].set_title('Density Profile')
        axs[0, 1].set_xlabel('Position')
        axs[0, 1].set_ylabel(r'${\rho}/{\rho_0}$ [-]')

        # plot temperature profile
        axs[1, 0].plot(T)
        axs[1, 0].grid()
        axs[1, 0].set_title('Temperature Profile')
        axs[1, 0].set_xlabel('Position')
        axs[1, 0].set_ylabel('$T/T_0$ [-]')

        # plot Mach profile
        M = V / (T**(1/2))
        axs[1, 1].plot(M)
        axs[1, 1].grid()
        axs[1, 1].set_title('Mach Profile')
        axs[1, 1].set_xlabel('Position')
        axs[1, 1].set_ylabel('$Mach$ [-]')

    def plot_residuals(self, residuals):
        """Plot the residuals of the calculations.
        This method generates a plot showing the evolution of residuals for density, velocity, and temperature over the iterations.
        It uses the residuals stored in the `self.residuals` dictionary, which is updated during the iterations.
        """
        fig = plt.figure(figsize=(12, 8))
        plt.title('Residuals', fontsize=16)
        if 'U1' in residuals:
            plt.plot(residuals['U1'], label='Conservative variable for mass')
            plt.plot(residuals['U2'], label='Conservative variable for momentum')
            plt.plot(residuals['U3'], label='Conservative variable for energy')
        else:
            plt.plot(residuals['rho'], label='Density')
            plt.plot(residuals['V'], label='Velocity')
            plt.plot(residuals['T'], label='Temperature')
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.yscale('log')
        plt.grid()
        plt.legend()

    def plot_contour(self, x, r, variable, variable_name, n_interpolated=1000):
        """Plot a contour plot of a variable along the nozzle.
        Parameters
        ----------
        x : np.ndarray
            Axial positions along the nozzle.
        r : np.ndarray
            Radius values of the nozzle.
        variable : np.ndarray
            Variable to be plotted (e.g., velocity, density, temperature).
        variable_name : str
            Name of the variable for labeling the plot.
        """
        # Interpolation des données pour une meilleure résolution sur l'affichage
        x_interpolated = np.linspace(x[0], x[-1], n_interpolated)
        variable_interpolated = np.interp(x_interpolated, x, variable)

        # Créer la grille Y symétrique autour de 0
        y = np.linspace(-max(r), max(r), n_interpolated)
        X, Y = np.meshgrid(x_interpolated, y)

        # 3. Masque pour rester dans la géométrie de la tuyère
        R_lin = interp1d(x, r, kind='linear')
        mask = np.abs(Y) <= R_lin(x_interpolated)

        # 4. Étendre le champ de Mach en 2D
        variable_2D = np.tile(variable_interpolated, (n_interpolated, 1))

        # Appliquer le masque
        variable_2D[~mask] = np.nan  # on met à NaN l'extérieur de la tuyère

        plt.figure(figsize=(12, 4))
        plt.contourf(X, Y, variable_2D, levels=50, cmap='plasma')
        plt.colorbar(label=variable_name)
        plt.title(f'Contour of {variable_name} along the nozzle')
        plt.xlabel(r'$x_{adim} [-]$')
        plt.ylabel(r'$r_{adim} [-]$')

