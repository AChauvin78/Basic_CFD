import numpy as np
import matplotlib.pyplot as plt

class Mac_Cormack:
    """
    A class to implement the MacCormack method for solving the compressible flow equations.

    Attributes
    ----------
    V0 : np.ndarray
        Initial velocity profile.
    rho0 : np.ndarray
        Initial density profile.
    T0 : np.ndarray
        Initial temperature profile.
    A : np.ndarray
        Area profile of the nozzle.
    delta_X : float
        Spatial step size.
    courant_number : float
        Courant number for stability.
    gamma : float
        Specific heat ratio (default is 1.4 for air).
    residuals : dict
        Dictionary to store residuals for density, velocity, and temperature.
    
    Methods
    -------
    drho_over_dt(V, rho, type='forward')
        Calculate the time derivative of density.
    dV_over_dt(V, rho, T, type='forward')
        Calculate the time derivative of velocity.
    dT_over_dt(V, T, type='forward')
        Calculate the time derivative of temperature.
    calculate_next_step_t(V, rho, T)
        Calculate the next time step for velocity, density, and temperature.
    calculate_delta_t(T, V)
        Calculate the time step based on the Courant condition.
    loop_over_iterations(n_ite)
        Loop over a specified number of iterations to solve the flow equations.
    plot_evolution_during_loop(V_for_all_ite, rho_for_all_ite, T_for_all_ite)
        Plot the evolution of velocity, density, and temperature during iterations.
    plot_final_state(V, rho, T)
        Plot the final state profiles of velocity, density, and temperature.
    plot_residuals()
        Plot the residuals of the calculations.
    """

    def __init__(self, V0, rho0, T0, A, delta_X, courant_number, gamma=1.4):
        """ Initialize the MacCormack method with given parameters.
        Parameters
        ----------
        V0 : np.ndarray
            Initial nondimensionalized velocity profile.
        rho0 : np.ndarray
            Initial nondimensionalized density profile.
        T0 : np.ndarray
            Initial nondimensionalized temperature profile.
        A : np.ndarray
            Nondimensionalized area profile of the nozzle.
        delta_X : float
            Spatial step size.
        courant_number : float
            Courant number for stability.
        gamma : float, optional
            Specific heat ratio (default is 1.4 for air).
        """
        self.V0 = V0
        self.rho0 = rho0
        self.T0 = T0
        self.A = A
        self.delta_X = delta_X
        self.courant_number = courant_number
        self.gamma = gamma
        self.residuals = {'rho': [], 'V': [], 'T': []}


    def drho_over_dt(self, V, rho, type='forward'):
        """Calculate the time derivative of density.
        Parameters
        ----------
        V : np.ndarray
            Velocity profile at the current time step.
        rho : np.ndarray
            Density profile at the current time step.
        type : str, optional
            Type of numerical derivative ('forward' or 'rearward', default is 'forward').
        Returns
        -------
        np.ndarray
            Time derivative of density.
        """
        delta_X = self.delta_X
        A = self.A
        i = list(range(1, len(V)-1))
        next_i = list(range(2, len(V)))
        previous_i = list(range(len(V)-2))
        if type == 'forward':
            drho_over_dt = -rho[i] * (V[next_i] - V[i])/delta_X - rho[i] * V[i] * (np.log(A[next_i]) - np.log(A[i]))/delta_X \
                            - V[i] * (rho[next_i] - rho[i])/delta_X
        elif type == 'rearward':
            drho_over_dt = -rho[i] * (V[i] - V[previous_i])/delta_X - rho[i] * V[i] * (np.log(A[i]) - np.log(A[previous_i]))/delta_X \
                            - V[i] * (rho[i] - rho[previous_i])/delta_X
        else:
            raise ValueError("type must be either 'forward' or 'rearward'")
        # Add ghost points to maintain the same length as the original array
        drho_over_dt = np.concatenate(([0], drho_over_dt, [0]))
        return drho_over_dt
    
    def dV_over_dt(self, V, rho, T, type='forward'):
        """Calculate the time derivative of velocity.
        Parameters
        ----------
        V : np.ndarray
            Velocity profile at the current time step.
        rho : np.ndarray
            Density profile at the current time step.
        T : np.ndarray
            Temperature profile at the current time step.
        type : str, optional
            Type of numerical derivative ('forward' or 'rearward', default is 'forward').
        Returns
        -------
        np.ndarray
            Time derivative of velocity.
        """
        delta_X = self.delta_X
        i = list(range(1, len(V)-1))
        next_i = list(range(2, len(V)))
        previous_i = list(range(len(V)-2))
        if type == 'forward':
            dV_over_dt = -V[i] * (V[next_i] - V[i])/delta_X - 1/self.gamma * (T[next_i] - T[i]) / delta_X \
                    - 1/self.gamma * T[i]/rho[i] * (rho[next_i] - rho[i])/delta_X
        elif type == 'rearward':
            dV_over_dt = -V[i] * (V[i] - V[previous_i])/delta_X - 1/self.gamma * (T[i] - T[previous_i]) / delta_X \
                    - 1/self.gamma * T[i]/rho[i] * (rho[i] - rho[previous_i])/delta_X
        else:
            raise ValueError("type must be either 'forward' or 'rearward'")
        # Add ghost points to maintain the same length as the original array   
        dV_over_dt = np.concatenate(([0], dV_over_dt, [0]))
        return dV_over_dt
    
    def dT_over_dt(self, V, T, type='forward'):
        """Calculate the time derivative of temperature.
        Parameters
        ----------
        V : np.ndarray
            Velocity profile at the current time step.
        T : np.ndarray
            Temperature profile at the current time step.
        type : str, optional
            Type of numerical derivative ('forward' or 'rearward', default is 'forward').
        Returns
        -------
        np.ndarray
            Time derivative of temperature.
        """
        delta_X = self.delta_X
        A = self.A
        i = list(range(1, len(V)-1))
        next_i = list(range(2, len(V)))
        previous_i = list(range(len(V)-2))
        if type == 'forward':
            dT_over_dt = -V[i] * (T[next_i] - T[i])/delta_X - (self.gamma - 1)*T[i]*((V[next_i] - V[i])/delta_X \
                    + V[i]*(np.log(A[next_i]) - np.log(A[i]))/delta_X)
        elif type == 'rearward':
            dT_over_dt = -V[i] * (T[i] - T[previous_i])/delta_X - (self.gamma - 1)*T[i]*((V[i] - V[previous_i])/delta_X \
                    + V[i]*(np.log(A[i]) - np.log(A[previous_i]))/delta_X)
        else:
            raise ValueError("type must be either 'forward' or 'rearward'")
        # Add ghost points to maintain the same length as the original array
        dT_over_dt = np.concatenate(([0], dT_over_dt, [0]))
        return dT_over_dt

    def calculate_next_step_t(self, V, rho, T):
        """Calculate the next time step for velocity, density, and temperature using the MacCormack method.
        Parameters
        ----------
        V : np.ndarray
            Velocity profile at the current time step.
        rho : np.ndarray
            Density profile at the current time step.
        T : np.ndarray
            Temperature profile at the current time step.
        Returns
        -------
        tuple
            V_next : np.ndarray
                Updated velocity profile after the next time step.
            rho_next : np.ndarray
                Updated density profile after the next time step.
            T_next : np.ndarray
                Updated temperature profile after the next time step.
        """
        
        # Calculate delta_t with the values at time t
        delta_t = self.calculate_delta_t(T, V)

        # Calculate time derivatives
        drho_over_dt = self.drho_over_dt(V, rho, type='forward')
        dV_over_dt = self.dV_over_dt(V, rho, T, type='forward')
        dT_over_dt = self.dT_over_dt(V, T, type='forward')

        # calculate predicted values of rho, V and T
        rho_predicted_next_step = rho + drho_over_dt*delta_t
        V_predicted_next_step = V + dV_over_dt*delta_t
        T_predicted_next_step = T + dT_over_dt*delta_t

        # corrector step
        drho_over_dt_predicted_next_step = self.drho_over_dt(V_predicted_next_step, rho_predicted_next_step, type='rearward')
        dV_over_dt_predicted_next_step = self.dV_over_dt(V_predicted_next_step, rho_predicted_next_step, T_predicted_next_step, type='rearward')
        dT_over_dt_predicted_next_step = self.dT_over_dt(V_predicted_next_step, T_predicted_next_step, type='rearward')

        # Average of time derivatives
        drho_over_dt_averaged = 1/2 * (drho_over_dt + drho_over_dt_predicted_next_step)
        dV_over_dt_averaged = 1/2 * (dV_over_dt + dV_over_dt_predicted_next_step)
        dT_over_dt_averaged = 1/2 * (dT_over_dt + dT_over_dt_predicted_next_step)

        self.residuals['rho'].append(np.max(np.abs(drho_over_dt_averaged)))
        self.residuals['V'].append(np.max(np.abs(dV_over_dt_averaged)))
        self.residuals['T'].append(np.max(np.abs(dT_over_dt_averaged)))

        # Corrected values of the flow field
        rho_corrected_next_step = rho + drho_over_dt_averaged*delta_t
        V_corrected_next_step = V + dV_over_dt_averaged*delta_t
        T_corrected_next_step = T + dT_over_dt_averaged*delta_t

        # Compute the variables at the boundaries
        # At the inlet, only the velocity is allowed to float, rho and T are constant
        V_corrected_next_step[0] = 2*V_corrected_next_step[1] - V_corrected_next_step[2]
        # At the outlet, all the variables are allowed to float
        rho_corrected_next_step[-1] = 2*rho_corrected_next_step[-2] - rho_corrected_next_step[-3]
        V_corrected_next_step[-1] = 2*V_corrected_next_step[-2] - V_corrected_next_step[-3]
        T_corrected_next_step[-1] = 2*T_corrected_next_step[-2] - T_corrected_next_step[-3]

        return V_corrected_next_step, rho_corrected_next_step, T_corrected_next_step
    
    def calculate_delta_t(self, T, V):
        """Calculate the time step based on the Courant condition.
        Parameters
        ----------
        T : np.ndarray
            Temperature profile at the current time step.
        V : np.ndarray
            Velocity profile at the current time step.
        Returns
        -------
        float
            Calculated time step based on the Courant condition.
        """
        delta_t_for_each_point = self.courant_number * self.delta_X / (T**(1/2) + V)
        return np.min(delta_t_for_each_point)

    def loop_over_iterations(self, n_ite):
        """Loop over a specified number of iterations to solve the flow equations.
        Parameters
        ----------
        n_ite : int
            Number of iterations to perform.
        Returns
        -------
        tuple
            V_for_all_ite : np.ndarray
                Array of velocity profiles at each iteration.
            rho_for_all_ite : np.ndarray
                Array of density profiles at each iteration.
            T_for_all_ite : np.ndarray
                Array of temperature profiles at each iteration.
        """
        # Initialize the variables to save the values of the intermediate iterations to plot the evolution
        V_for_all_ite = np.zeros((n_ite, len(V0)))
        rho_for_all_ite = np.zeros((n_ite, len(rho0)))
        T_for_all_ite = np.zeros((n_ite, len(T0)))
        V_for_all_ite[0] = V0
        rho_for_all_ite[0] = rho0
        T_for_all_ite[0] = T0

        # Main loop
        for i in range(n_ite-1):
            V, rho, T = self.calculate_next_step_t(V_for_all_ite[i], rho_for_all_ite[i], T_for_all_ite[i])
            V_for_all_ite[i+1] = V
            rho_for_all_ite[i+1] = rho
            T_for_all_ite[i+1] = T

        return V_for_all_ite, rho_for_all_ite, T_for_all_ite
        

    def plot_evolution_during_loop(self, V_for_all_ite, rho_for_all_ite, T_for_all_ite):
        """Plot the evolution of velocity, density, and temperature during iterations.
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

    def plot_residuals(self):
        """Plot the residuals of the calculations.
        This method generates a plot showing the evolution of residuals for density, velocity, and temperature over the iterations.
        It uses the residuals stored in the `self.residuals` dictionary, which is updated during the iterations.
        """
        residuals = self.residuals
        fig = plt.figure(figsize=(12, 8))
        plt.title('Residuals', fontsize=16)
        plt.plot(residuals['rho'], label='Density')
        plt.plot(residuals['V'], label='Velocity')
        plt.plot(residuals['T'], label='Temperature')
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.yscale('log')
        plt.grid()
        plt.legend()

if __name__ == '__main__':
    x = np.linspace(0, 3, 31)
    delta_X = x[1] - x[0]
    A = 1 + 2.2*(x-1.5)**2
    rho0 = 1 - 0.3146*x
    T0 = 1 - 0.2314*x
    V0 = (0.1 + 1.09*x)*T0**(1/2)
    courant_number = 0.5

    mac_cormack = Mac_Cormack(V0, rho0, T0, A, delta_X, courant_number)
    V_for_all_ite, rho_for_all_ite, T_for_all_ite = mac_cormack.loop_over_iterations(1400)
    mac_cormack.plot_evolution_during_loop(V_for_all_ite, rho_for_all_ite, T_for_all_ite)
    mac_cormack.plot_final_state(V_for_all_ite[-1], rho_for_all_ite[-1], T_for_all_ite[-1])
    mac_cormack.plot_residuals()

    plt.tight_layout()
    plt.show()
