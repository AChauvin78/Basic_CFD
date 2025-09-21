from Mac_Cormack import Mac_Cormack
from Mac_Cormack_conservation_form import Mac_Cormack_Conservation_Form
from Nozzle import Nozzle
from Plot import Plot
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("Quasi-1D Nozzle Simulation")

st.subheader("Define the nozzle parameters")

# Define the nozzle parameters
length = st.slider("Length", 1.0, 5.0, 3.0, 0.1)
coeff_conv_div = st.slider("coeff_conv_div", 0.1, 5.0, 2.2, 0.1)

# Checkbox
use_coeff_conv_div_after_throat = st.checkbox("définir coeff_conv_div_after_throat ?", value=False)

if use_coeff_conv_div_after_throat:
    coeff_conv_div_after_throat = st.slider("coeff_conv_div_after_throat", 0.01, 1.0, 0.2223, 0.01)
else:
    coeff_conv_div_after_throat = None
discretization_points = st.slider("Discretization points", 10, 200, 61)

nozzle = Nozzle(length=length, coeff_conv_div=coeff_conv_div, coeff_conv_div_after_throat=coeff_conv_div_after_throat, 
                discretization_points=discretization_points)
x, A = nozzle.discretize()
r = nozzle.get_radius(x)
delta_X = x[1] - x[0]

# Plot the nozzle profile
nozzle.plot_nozzle_profile()


st.subheader("Define the simulation parameters")

courant_number = st.slider("courant_number", 0.0, 1.5, 0.5, 0.1)
supersonic_exit = st.checkbox("Supersonic outlet ?", value=True)
if not supersonic_exit:
    pe = st.slider("Outlet pressure Pe", 0.0, 1.0, 0.93, 0.01)
    Cx = st.slider("Artificial viscosity coefficient Cx", 0.0, 1.0, 0.2, 0.1)
else:
    pe = None
    Cx = 0.0

number_of_iterations = st.slider("number_of_iterations", 100, 10000, 2000, 100)

# Initial conditions
rho0 = 1 + 0.0*x
T0 = 1 + 0.0*x
V0 = (0.1 + 0.11*x)

mac_cormack = Mac_Cormack_Conservation_Form(V0, rho0, T0, A, delta_X, courant_number, Cx=Cx, supersonic_exit=supersonic_exit, pe=pe)


if st.button("Launch the simulation"):

    U1_for_all_ite, U2_for_all_ite, U3_for_all_ite, residuals = mac_cormack.loop_over_iterations(number_of_iterations)
    print(f"U1 : {U1_for_all_ite[-1]}")
    print(f"U2 : {U2_for_all_ite[-1]}")
    print(f"U3 : {U3_for_all_ite[-1]}")

    V_for_all_ite, rho_for_all_ite, T_for_all_ite = mac_cormack.convert_U_to_primitive(U1_for_all_ite, U2_for_all_ite, U3_for_all_ite)

    print(f"V : {V_for_all_ite[-1]}")
    print(f"rho : {rho_for_all_ite[-1]}")
    print(f"T : {T_for_all_ite[-1]}")

    st.success("Simulation completed!")

    plot = Plot()
    Mach = V_for_all_ite[-1] / (T_for_all_ite[-1]**(1/2))
    plot.plot_contour(x, r, Mach, 'Mach')
    plot.plot_final_state(V_for_all_ite[-1], rho_for_all_ite[-1], T_for_all_ite[-1])
    plot.plot_residuals(residuals)
    plot.plot_evolution_during_loop(V_for_all_ite, rho_for_all_ite, T_for_all_ite)
    