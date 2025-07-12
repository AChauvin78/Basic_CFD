import numpy as np
from matplotlib import pyplot as plt

class Nozzle:
    """
    A class to represent a nozzle with a variable cross-sectional area.
    The dimensions are nondimensionalized. The area of the throat is 1.

    Attributes
    ----------
    length : float
        The length of the nozzle.
    coeff_conv_div : float
        Coefficient for the conversion/divergence of the nozzle area.
    discretization_points : int
        Number of points to discretize the nozzle for calculations.
    
    Methods
    -------
    get_area(x)
        Calculate the area of the nozzle at position x.
    get_radius(x)
        Calculate the radius of the nozzle at position x.
    discretize()
        Discretize the nozzle into a specified number of points for calculation.
    plot_nozzle_profile(num_points=100)
        Plot the nozzle profile based on the calculated areas and radii.
    """

    def __init__(self, length, coeff_conv_div, discretization_points=100):
        """ Initialize the Nozzle with given parameters.
        Parameters 
        ----------
        length : float
            The length of the nozzle.
        coeff_conv_div : float
            Coefficient for the conversion/divergence of the nozzle area.
        discretization_points : int
            Number of points to discretize the nozzle for calculations.
        """
            
        self.length = length
        self.coeff_conv_div = coeff_conv_div
        self.discretization_points = discretization_points

    def get_area(self, x):
        """Calculate the area of the nozzle at position x.
        
        Parameters
        ----------
        x : float
            The position along the nozzle length where the area is calculated.
        
        Returns
        -------        
        float
            The area of the nozzle at position x.

        Raises
        ------  
        ValueError
            If x is outside the range [0, length].
        """
        if np.any((x < 0) | (x > self.length)):
            raise ValueError(f"x must be within the range [0, {self.length}]")
        return 1 + self.coeff_conv_div * (x - self.length / 2) ** 2
    
    def get_radius(self, x):
        """
        Calculate the radius of the nozzle at a given axial position.

        Parameters:
            x (float): The axial position along the nozzle where the radius is to be calculated.

        Returns:
            float: The radius of the nozzle at position x.

        Notes:
            This method computes the cross-sectional area at position x using `self.get_area(x)`,
            then calculates the corresponding radius assuming a circular cross-section.
        """
        """Calculate the radius of the nozzle at position x."""
        area = self.get_area(x)
        return np.sqrt(area / np.pi)

    def discretize(self):
        """
        Discretizes the nozzle along its length into a specified number of points.

        Returns:
            tuple:
                - x_values (numpy.ndarray): 1D array of positions along the nozzle from 0 to its total length.
                - areas (list of float): List of cross-sectional areas at each position in x_values.

        Notes:
            The number of discretization points is determined by the 'discretization_points' attribute.
            The cross-sectional area at each position is computed using the 'get_area' method.
        """
        x_values = np.linspace(0, self.length, self.discretization_points)
        areas = np.array([self.get_area(x) for x in x_values])
        return x_values, areas

    def plot_nozzle_profile(self, num_points=100):
        """
        Plots the nozzle profile along its length.
        This method generates a plot of the nozzle's radius as a function of its length, displaying both the upper and lower boundaries (symmetric about the centerline). It also marks the discretized points used for calculations along the nozzle axis.
        Parameters:
            num_points (int, optional): Number of points to use for plotting the smooth nozzle profile. Default is 100.
        Notes:
            - The method uses matplotlib for plotting.
            - The discretized points are shown as red 'x' markers along the centerline.
            - The nozzle profile is plotted as a symmetric shape about the centerline.
        """
        x_values = np.linspace(0, self.length, num_points)
        y_values = np.array([self.get_radius(x) for x in x_values])

        # Discretized points used for calculation
        x_values_discretized = np.linspace(0, self.length, self.discretization_points)

        plt.figure(figsize=(10, 5))
        plt.plot(x_values, y_values, color='C0', label='Nozzle Profile')
        plt.plot(x_values, -y_values, color='C0')
        plt.plot(x_values_discretized, np.zeros(self.discretization_points), label='Discretized Points\nused for calculation', marker='x', linestyle='--', color='red')
        plt.title('Nozzle Profile')
        plt.xlabel('Length [-]')
        plt.ylabel('Radius [-]')
        plt.grid()
        plt.legend()
        plt.show()


# Example usage
if __name__ == "__main__":
    nozzle = Nozzle(length=3, coeff_conv_div=2.2, discretization_points=25)
    nozzle.plot_nozzle_profile()
    
    # Get area and radius at a specific point
    x = 2
    print(f"Area at x={x}: {nozzle.get_area(x)}")
    print(f"Radius at x={x}: {nozzle.get_radius(x)}")