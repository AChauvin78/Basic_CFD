import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# st.title("Simulation de tuyère")

# P0 = st.number_input("Pression amont (Pa)", value=101325)
# Pe = st.number_input("Pression sortie (Pa)", value=50000)
# N  = st.slider("Nombre de mailles", 10, 200, 50)

# if st.button("Lancer la simulation"):
#     # Exemple : profil de Mach fictif
#     x = np.linspace(0,1,N)
#     Mach = 1 + 0.5*np.sin(2*np.pi*x)
    
#     fig, ax = plt.subplots()
#     ax.plot(x, Mach)
#     ax.set_xlabel("x")
#     ax.set_ylabel("Mach")
#     st.pyplot(fig)


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

freq = st.slider("Fréquence", 0.1, 5.0, 1.0, 0.1)

x = np.linspace(0, 2*np.pi, 200)
y = np.sin(freq * x)

fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)