# Breaking Bloch's Theorem
_Authors:_ Victor Klippgen & Olav Førland

Repository for the code to the project "Breaking Bloch's Theorem" in the course AM207: Advanced Scientific Computing at Harvard. 
The structure is as follows:

```plaintext
├── experiments                         # Folder containing experiments
│   ├── band_gap.ipynb
│   ├── displacement.ipynb
│   └── electronic_structure.ipynb
├── src                                 # Source code
│   ├── __init__.py
│   ├── constants.py                    # Constants used across the project
│   ├── simulation.py                   # Monte Carlo Simulations
│   └── utils.py                        # Utility functions
├── visualize                           # Visualization methods
│   ├── __init__.py
│   ├── band_gap.py                     # Visualization for band gaps
│   ├── displacement.py                 # Visualization for displacement
│   └── electronic_structure.py         # Visualization for electronic structure
```

### Experiments

All experiments are found as notebooks in the `experiments` folder. To run them you need the following packages:

- `numpy`
- `scipy`
- `matplotlib`