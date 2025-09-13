# SUPERSONIC TEST CASE

# Get the results from the solver

from Nozzle import Nozzle
from Mac_Cormack import Mac_Cormack
import pandas as pd

nozzle = Nozzle(length=3, coeff_conv_div=2.2, discretization_points=31)
x, A = nozzle.discretize()
r = nozzle.get_radius(x)

delta_X = x[1] - x[0]
rho0 = 1 - 0.3146*x
T0 = 1 - 0.2314*x
V0 = (0.1 + 1.09*x)*T0**(1/2)
courant_number = 0.5

mac_cormack = Mac_Cormack(V0, rho0, T0, A, delta_X, courant_number)
V_for_all_ite, rho_for_all_ite, T_for_all_ite = mac_cormack.loop_over_iterations(1000)

df_sim = pd.DataFrame({
    "I": list(range(1, 32)),
    "x/L": x,
    "A/A*": A,
    "rho/rho0": rho_for_all_ite[-1],
    "V/a0": V_for_all_ite[-1],
    "T/T0": T_for_all_ite[-1]
})

# mac_cormack.plot_evolution_during_loop(V_for_all_ite, rho_for_all_ite, T_for_all_ite)
# mac_cormack.plot_final_state(V_for_all_ite[-1], rho_for_all_ite[-1], T_for_all_ite[-1])
# mac_cormack.plot_residuals()
# Mach = V_for_all_ite[-1] / (T_for_all_ite[-1]**(1/2))
# mac_cormack.plot_contour(x, r, Mach, 'Mach')

# Compare to the values from the book
data = {
    "I": list(range(1, 32)),
    "x/L": [
        0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900,
        1.000, 1.100, 1.200, 1.300, 1.400, 1.500, 1.600, 1.700, 1.800, 1.900,
        2.000, 2.100, 2.200, 2.300, 2.400, 2.500, 2.600, 2.700, 2.800, 2.900,
        3.000
    ],
    "A/A*": [
        5.950, 5.312, 4.718, 4.168, 3.662, 3.200, 2.782, 2.408, 2.078, 1.792,
        1.550, 1.352, 1.198, 1.088, 1.022, 1.000, 1.022, 1.088, 1.198, 1.352,
        1.550, 1.792, 2.078, 2.408, 2.782, 3.200, 3.662, 4.168, 4.718, 5.312,
        5.950
    ],
    "rho/rho0": [
        1.000, 0.998, 0.997, 0.994, 0.992, 0.987, 0.982, 0.974, 0.963, 0.947,
        0.924, 0.892, 0.849, 0.792, 0.721, 0.639, 0.551, 0.465, 0.386, 0.318,
        0.262, 0.216, 0.179, 0.150, 0.126, 0.107, 0.092, 0.080, 0.069, 0.061,
        0.053
    ],
    "V/a0": [
        0.099, 0.112, 0.125, 0.143, 0.162, 0.187, 0.215, 0.251, 0.294, 0.346,
        0.409, 0.485, 0.575, 0.678, 0.793, 0.914, 1.037, 1.155, 1.263, 1.361,
        1.446, 1.519, 1.582, 1.636, 1.683, 1.723, 1.759, 1.789, 1.817, 1.839,
        1.862
    ],
    "T/T0": [
        1.000, 0.999, 0.999, 0.998, 0.997, 0.995, 0.993, 0.989, 0.985, 0.978,
        0.969, 0.956, 0.937, 0.911, 0.878, 0.836, 0.789, 0.737, 0.684, 0.633,
        0.588, 0.541, 0.502, 0.467, 0.436, 0.408, 0.384, 0.362, 0.342, 0.325,
        0.308
    ]
}

df_ref = pd.DataFrame(data)

pd.set_option('display.precision', 3)
print("Reference Data:")
print(df_ref)
print("\nSimulation Data:")
print(df_sim)
print("\nDifference (Sim - Ref):")
print(df_sim - df_ref)

# test si une valeur du df est > 5e-3
condition = (abs(df_sim - df_ref) > 5e-3).any().any()
print(f"Test passed : {not condition}")   # False si au moins une valeur > 5e-3, sinon True



# SUBSONIC TEST CASE
# Get the results from the solver
from Nozzle import Nozzle
from Mac_Cormack import Mac_Cormack
import pandas as pd

nozzle = Nozzle(length=3, coeff_conv_div=2.2, coeff_conv_div_after_throat=0.2223, discretization_points=31)
x, A = nozzle.discretize()
r = nozzle.get_radius(x)

delta_X = x[1] - x[0]
rho0 = 1-0.023*x
T0 = 1-0.00933*x
V0 = 0.05 + 0.11*x
pe = 0.93
courant_number = 0.5

mac_cormack = Mac_Cormack(V0, rho0, T0, A, delta_X, courant_number, supersonic=False, pe=pe)
V_for_all_ite, rho_for_all_ite, T_for_all_ite = mac_cormack.loop_over_iterations(5000)

from matplotlib import pyplot as plt
mac_cormack.plot_evolution_during_loop(V_for_all_ite, rho_for_all_ite, T_for_all_ite)
mac_cormack.plot_final_state(V_for_all_ite[-1], rho_for_all_ite[-1], T_for_all_ite[-1])
mac_cormack.plot_residuals()
Mach = V_for_all_ite[-1] / (T_for_all_ite[-1]**(1/2))
mac_cormack.plot_contour(x, r, Mach, 'Mach')
plt.show()

df_sim = pd.DataFrame({
    "I": list(range(1, 32)),
    "x/L": x,
    "A/A*": A,
    "rho/rho0": rho_for_all_ite[-1],
    "V/a0": V_for_all_ite[-1],
    "T/T0": T_for_all_ite[-1]
})

