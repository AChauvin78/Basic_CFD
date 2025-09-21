Basic_CFD

Quasi-1D CFD Solver for Compressible Flow in a Nozzle

This project implements a Python-based CFD solver to simulate compressible fluid flow through a nozzle. The solver uses the MacCormack method in both conservative and non-conservative forms. The latest version provides a Streamlit interface to select options, run simulations, and display results interactively.

🔧 Features

- MacCormack Solver: Explicit solver for quasi-1D compressible flows.
- Conservative & Non-Conservative Forms: Two solver options (Mac_Cormack.py and Mac_Cormack_conservation_form.py).
- Nozzle Geometry: Define nozzle shape and boundary conditions via Nozzle.py.
- Visualization: Graphical output of velocity, pressure, density, and temperature profiles (Plot.py).
- Interactive GUI: Run simulations and visualize results using Streamlit (main.py).
- Unit Tests: Validate the solver through predefined test cases (Test.py).

📁 Repository Structure
Basic_CFD/
├─ Main.py                            # Streamlit interface to run simulations and view plots
├─ Mac_Cormack.py                      # Standard MacCormack solver
├─ Mac_Cormack_conservation_form.py    # Conservative form MacCormack solver
├─ Nozzle.py                           # Nozzle geometry and boundary conditions
├─ Plot.py                             # Plotting functions
├─ Test.py                             # Unit tests
└─ __pycache__/                        # Compiled Python files

🚀 Installation

Clone the repository:
git clone https://github.com/AChauvin78/Basic_CFD.git
cd Basic_CFD


Install dependencies:
pip install numpy matplotlib streamlit

📊 Usage
Using the Streamlit GUI

Run the main interface:

streamlit run main.py


The GUI allows you to:
- Select solver options (conservative or non-conservative).
- Configure nozzle geometry and boundary conditions.
- Run simulations interactively.
- Display plots directly in the interface.
- Running the Solvers Directly

Non-conservative solver:
python Mac_Cormack.py

Conservative solver:
python Mac_Cormack_conservation_form.py

📈 Visualization

The Plot.py module generates plots for:
- Velocity
- Pressure
- Density
- Temperature
