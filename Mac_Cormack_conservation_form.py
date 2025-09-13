import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Mac_Cormack_Conservation_Form:
    """
    A class to implement the MacCormack method for solving the compressible flow equations using the conservation form.

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

    def __init__(self, V0, rho0, T0, A, delta_X, courant_number, supersonic=True, pe=None, gamma=1.4):
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
        self.T0 = T0
        self.U1_0 = rho0 * A  # Initial mass
        self.U2_0 = V0 * rho0 * A # Initial momentum
        self.U3_0 = rho0 * (T0 / (gamma - 1) + gamma/2 * V0**2) * A
        self.A = A
        self.delta_X = delta_X
        self.courant_number = courant_number
        self.supersonic = supersonic
        self.pe = pe
        self.gamma = gamma
        self.residuals = {'U1': [], 'U2': [], 'U3': []}

    def rho(self, U1, A):
        """Calculate the density from the conservative variable U1 and area A.
        Parameters
        ----------
        U1 : np.ndarray
            Conservative variable for mass (rho * A).
        A : np.ndarray
            Area profile of the nozzle.
        Returns
        -------
        np.ndarray
            Density profile.
        """
        return U1 / A
    
    def V(self, U1, U2):
        """Calculate the velocity from the conservative variables U1 and U2.
        Parameters
        ----------
        U1 : np.ndarray
            Conservative variable for mass (rho * A).
        U2 : np.ndarray
            Conservative variable for momentum (rho * V * A).
        Returns
        -------
        np.ndarray
            Velocity profile.
        """
        return U2 / U1
    
    def T(self, U1, U2, U3):
        """Calculate the temperature from the conservative variables U1, U2, and U3.
        Parameters
        ----------
        U1 : np.ndarray
            Conservative variable for mass (rho * A).
        U2 : np.ndarray
            Conservative variable for momentum (rho * V * A).
        U3 : np.ndarray
            Conservative variable for energy (rho * (T/(gamma-1) + gamma/2 * V^2) * A).
        gamma : float
            Specific heat ratio.
        Returns
        -------
        np.ndarray
            Temperature profile.
        """
        return (U3 / U1 - self.gamma/2 * (U2 / U1)**2) * (self.gamma - 1)
    
    def p(self, U1, U2, U3):
        """Calculate the pressure from the conservative variables U1, U2, and U3 using p = rho*T.
        Parameters
        ----------
        U1 : np.ndarray
            Conservative variable for mass (rho * A).
        U2 : np.ndarray
            Conservative variable for momentum (rho * V * A).
        U3 : np.ndarray
            Conservative variable for energy (rho * (T/(gamma-1) + gamma/2 * V^2) * A).
        gamma : float
            Specific heat ratio.
        Returns
        -------
        np.ndarray
            Pressure profile.
        """
        return (U3 / U1 - self.gamma/2 * (U2 / U1)**2) * (self.gamma - 1) * (U1 / self.A)

    def F1(self, U2):
        """Calculate the flux F1 for the mass conservation equation.
        Parameters
        ----------
        U1 : np.ndarray
            Conservative variable for mass (rho * A).
        U2 : np.ndarray
            Conservative variable for momentum (rho * V * A).
        Returns
        -------
        np.ndarray
            Flux F1 for the mass conservation equation.
        """
        return U2
    
    def F2(self, U1, U2, U3):
        """Calculate the flux F2 for the momentum conservation equation.
        Parameters
        ----------
        U1 : np.ndarray
            Conservative variable for mass (rho * A).
        U2 : np.ndarray
            Conservative variable for momentum (rho * V * A).
        U3 : np.ndarray
            Conservative variable for energy (rho * (T/(gamma-1) + gamma/2 * V^2) * A).
        Returns
        -------
        np.ndarray
            Flux F2 for the momentum conservation equation.
        """
        return U2**2 / U1 + (self.gamma - 1)/ self.gamma * (U3 - self.gamma/2 * U2**2 / U1)
    
    def F3(self, U1, U2, U3):
        """Calculate the flux F3 for the energy conservation equation.
        Parameters
        ----------
        U1 : np.ndarray
            Conservative variable for mass (rho * A).
        U2 : np.ndarray
            Conservative variable for momentum (rho * V * A).
        U3 : np.ndarray
            Conservative variable for energy (rho * (T/(gamma-1) + gamma/2 * V^2) * A).
        Returns
        -------
        np.ndarray
            Flux F3 for the energy conservation equation.
        """
        return self.gamma * U2 * U3 / U1 - self.gamma * (self.gamma - 1) / 2 * U2**3 / U1**2
    
    def J2(self, U1, U2, U3):
        """Calculate the source term J2 for the momentum conservation equation.
        Parameters
        ----------
        U1 : np.ndarray
            Conservative variable for mass (rho * A).
        U2 : np.ndarray
            Conservative variable for momentum (rho * V * A).
        U3 : np.ndarray
            Conservative variable for energy (rho * (T/(gamma-1) + gamma/2 * V^2) * A).
        Returns
        -------
        np.ndarray
            Source term J2 for the momentum conservation equation.
        """
        return (self.gamma - 1) / self.gamma * (U3 - self.gamma/2 * U2**2 / U1) * np.gradient(np.log(self.A), self.delta_X)

    def dU_over_dt(self, U, F, J, type='forward'):
        """Calculate the time derivative of the variable U using the flux F and source J.
        Parameters
        ----------
        U : np.ndarray
            The conservative variable for which to calculate the time derivative.
        F : np.ndarray
            The flux used to calculate the derivative.
        J : np.ndarray
            The source term used in the calculation.
        type : str, optional
            Type of numerical derivative ('forward' or 'rearward', default is 'forward').
        Returns
        -------
        np.ndarray
            Time derivative of the conservative variable U.
        """
        delta_X = self.delta_X
        i = list(range(1, len(U)-1))
        next_i = list(range(2, len(U)))
        previous_i = list(range(len(U)-2))
        if type == 'forward':
            dU_over_dt = - (F[next_i] - F[i])/delta_X + J[i]
        elif type == 'rearward':
            dU_over_dt = - (F[i] - F[previous_i])/delta_X + J[i]
        else:
            raise ValueError("type must be either 'forward' or 'rearward'")
        # Add ghost points to maintain the same length as the original array
        dU_over_dt = np.concatenate(([0], dU_over_dt, [0]))
        return dU_over_dt

    def calculate_next_step_t(self, U1, U2, U3):
        """Calculate the next time step for velocity, density, and temperature using the MacCormack method.
        Parameters
        ----------
        U1 : np.ndarray
            Conservative variable for mass (rho * A) at the current time step.
        U2 : np.ndarray
            Conservative variable for momentum (rho * V * A) at the current time step.
        U3 : np.ndarray
            Conservative variable for energy (rho * (T/(gamma-1) + gamma/2 * V^2) * A) at the current time step.
        Returns
        -------
        tuple
            U1_corrected_next_step : np.ndarray
                Conservative variable for mass at the next time step.
            U2_corrected_next_step : np.ndarray
                Conservative variable for momentum at the next time step.
            U3_corrected_next_step : np.ndarray
                Conservative variable for energy at the next time step.
        """
        
        # Calculate delta_t with the values at time t
        delta_t = self.calculate_delta_t(U1, U2, U3)

        # Calculate the flux and source terms
        F1 = self.F1(U2)
        F2 = self.F2(U1, U2, U3)
        F3 = self.F3(U1, U2, U3)
        J2 = self.J2(U1, U2, U3)

        # Calculate time derivatives
        dU1_over_dt = self.dU_over_dt(U1, F1, 0*U1, type='forward')
        dU2_over_dt = self.dU_over_dt(U2, F2, J2, type='forward')
        dU3_over_dt = self.dU_over_dt(U3, F3, 0*U3, type='forward')

        # calculate predicted values of rho, V and T
        U1_predicted_next_step = U1 + dU1_over_dt*delta_t
        U2_predicted_next_step = U2 + dU2_over_dt*delta_t
        U3_predicted_next_step = U3 + dU3_over_dt*delta_t

        # calculate predicted fluxes and source terms
        F1_predicted_next_step = self.F1(U2_predicted_next_step)
        F2_predicted_next_step = self.F2(U1_predicted_next_step, U2_predicted_next_step, U3_predicted_next_step)
        F3_predicted_next_step = self.F3(U1_predicted_next_step, U2_predicted_next_step, U3_predicted_next_step)
        J2_predicted_next_step = self.J2(U1_predicted_next_step, U2_predicted_next_step, U3_predicted_next_step)

        # corrector step
        dU1_over_dt_predicted_next_step = self.dU_over_dt(U1_predicted_next_step, F1_predicted_next_step, 0*U1_predicted_next_step, type='rearward')
        dU2_over_dt_predicted_next_step = self.dU_over_dt(U2_predicted_next_step, F2_predicted_next_step, J2_predicted_next_step, type='rearward')
        dU3_over_dt_predicted_next_step = self.dU_over_dt(U3_predicted_next_step, F3_predicted_next_step, 0*U3_predicted_next_step, type='rearward')

        # Average of time derivatives
        dU1_over_dt_averaged = 1/2 * (dU1_over_dt + dU1_over_dt_predicted_next_step)
        dU2_over_dt_averaged = 1/2 * (dU2_over_dt + dU2_over_dt_predicted_next_step)
        dU3_over_dt_averaged = 1/2 * (dU3_over_dt + dU3_over_dt_predicted_next_step)

        self.residuals['U1'].append(np.max(np.abs(dU1_over_dt_averaged)))
        self.residuals['U2'].append(np.max(np.abs(dU2_over_dt_averaged)))
        self.residuals['U3'].append(np.max(np.abs(dU3_over_dt_averaged)))

        # Corrected values of the flow field
        U1_corrected_next_step = U1 + dU1_over_dt_averaged*delta_t
        U2_corrected_next_step = U2 + dU2_over_dt_averaged*delta_t
        U3_corrected_next_step = U3 + dU3_over_dt_averaged*delta_t

        # Compute the variables at the boundaries
        # At the inlet, only the velocity is allowed to float, rho and T are constant
        # Therefore, U2 and U3 are allowed to float and U1 is imposed
        U2_corrected_next_step[0] = 2*U2_corrected_next_step[1] - U2_corrected_next_step[2]
        U3_corrected_next_step[0] = U1_corrected_next_step[0] * (self.T0[0] / (self.gamma - 1) + self.gamma/2 * (U2_corrected_next_step[0]/U1_corrected_next_step[0])**2)
        # At the outlet, all the variables are allowed to float when supersonic
        if self.supersonic:
            U1_corrected_next_step[-1] = 2*U1_corrected_next_step[-2] - U1_corrected_next_step[-3]
            U2_corrected_next_step[-1] = 2*U2_corrected_next_step[-2] - U2_corrected_next_step[-3]
            U3_corrected_next_step[-1] = 2*U3_corrected_next_step[-2] - U3_corrected_next_step[-3]
        else: # if subsonic, impose the static pressure at the exit
            # rho is chosen to float, T is imposed from p = rho*T and V is allowed to float
            # Therefore U1 and U2 are allowed to float and U3 needs to be imposed
            U1_corrected_next_step[-1] = 2*U1_corrected_next_step[-2] - U1_corrected_next_step[-3]
            U2_corrected_next_step[-1] = 2*U2_corrected_next_step[-2] - U2_corrected_next_step[-3]
            rho_exit = U1_corrected_next_step[-1] / self.A[-1]
            T_exit = self.pe / rho_exit
            U3_corrected_next_step[-1] = U1_corrected_next_step[-1] * (T_exit / (self.gamma - 1) + self.gamma/2 * (U2_corrected_next_step[-1]/U1_corrected_next_step[-1])**2)

        return U1_corrected_next_step, U2_corrected_next_step, U3_corrected_next_step
    
    def calculate_delta_t(self, U1, U2, U3):
        """Calculate the time step based on the Courant condition.
        Use the minimum of the time steps calculated for each point along the nozzle.

        Parameters
        ----------
        U1 : np.ndarray
            Conservative variable for mass (rho * A).
        U2 : np.ndarray
            Conservative variable for momentum (rho * V * A).
        U3 : np.ndarray
            Conservative variable for energy (rho * (T/(gamma-1) + gamma/2 * V^2) * A).
        Returns
        -------
        float
            Calculated time step based on the Courant condition.
        """
        T = self.T(U1, U2, U3)
        V = self.V(U1, U2)
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
        U1_for_all_ite = np.zeros((n_ite, len(self.U1_0)))
        U2_for_all_ite = np.zeros((n_ite, len(self.U2_0)))
        U3_for_all_ite = np.zeros((n_ite, len(self.U3_0)))
        U1_for_all_ite[0] = self.U1_0
        U2_for_all_ite[0] = self.U2_0
        U3_for_all_ite[0] = self.U3_0

        # Main loop
        for i in range(n_ite-1):
            U1, U2, U3 = self.calculate_next_step_t(U1_for_all_ite[i], U2_for_all_ite[i], U3_for_all_ite[i])
            U1_for_all_ite[i+1] = U1
            U2_for_all_ite[i+1] = U2
            U3_for_all_ite[i+1] = U3

        return U1_for_all_ite, U2_for_all_ite, U3_for_all_ite, self.residuals
        
    def convert_U_to_primitive(self, U1, U2, U3):
        """Convert conservative variables U1, U2, U3 to primitive variables V, rho, T.
        Parameters
        ----------
        U1 : np.ndarray
            Conservative variable for mass (rho * A).
        U2 : np.ndarray
            Conservative variable for momentum (rho * V * A).
        U3 : np.ndarray
            Conservative variable for energy (rho * (T/(gamma-1) + gamma/2 * V^2) * A).
        Returns
        -------
        tuple
            V : np.ndarray
                Velocity profile.
            rho : np.ndarray
                Density profile.
            T : np.ndarray
                Temperature profile.
        """
        rho = self.rho(U1, self.A)
        V = self.V(U1, U2)
        T = self.T(U1, U2, U3)
        return V, rho, T