import numpy as np

from .constants import HBAR, HBYK  # Planck's const / Boltzmann const
from .constants import a, m, k0, beta, scale_factor

def sample_phonon_number(omega, T):
    """
    Perform inverse sampling from the Bose-Einstein distribution to compute the phonon number.

    Parameters:
        omega (float): Phonon frequency (rad/s).
        T (float): Temperature (Kelvin). If T is 0, the function returns 0 to avoid division errors.

    Returns:
        int: Sampled phonon number.
    """
    if T == 0:  # Skip if T = 0 to avoid inf error
        return 0
    exponent = HBYK * omega / T
    p = np.exp(-exponent)
    u = np.random.uniform(0, 1)
    
    # Inverse transform sampling for geometric distribution shifted by 1
    n_real = np.log(u) / np.log(p)
    n = max(int(np.floor(n_real)), 0)
    return n

def get_wavevector_1d(N, a=a):
    """
    Generate a 1D wavevector for a lattice of N points.

    Parameters:
        N (int): Number of points in the lattice.
        a (float, optional): Lattice constant. Default is imported from `constants`.

    Returns:
        np.ndarray: Array of wavevector values.
    """
    return np.linspace(0, 2 * np.pi / (N * a), N)

def calculate_phonon_frequency(N, m=m, k0=k0):
    """
    Calculate the phonon frequency for a 1D chain of N atoms.

    Parameters:
        N (int): Number of atoms in the chain.
        m (float, optional): Atomic mass. Default is imported from `constants`.
        k0 (float, optional): Spring constant. Default is imported from `constants`.

    Returns:
        np.ndarray: Array of phonon frequencies.
    """
    return 2 * np.sqrt(k0 / m) * np.sin(np.pi * np.arange(N) / N)

def calculate_hopping_scale_factor(u, beta, scale_factor):
    """
    Compute modulated hopping parameters based on atomic displacements.

    Parameters:
        u (np.ndarray): Array of atomic displacements.
        beta (float): Decay constant for hopping modulation.
        scale_factor (float): Scaling factor for displacements.

    Returns:
        np.ndarray: Array of hopping scale factors.
    """
    N = len(u)
    t_scale = np.zeros(N)
    
    for i in range(N):
        if i == N - 1:
            # Periodic boundary condition
            d = -u[i] + u[0]
        else:
            d = u[i + 1] - u[i]
        
        t_scale[i] = np.exp(-beta * d * scale_factor)

    return t_scale

def construct_hamiltonian(N, t0, t1, t_scale):
    """
    Construct the electronic Hamiltonian with modulated hopping parameters.

    Parameters:
        N (int): Size of the Hamiltonian matrix (number of sites).
        t0 (float): Hopping parameter for even-indexed bonds.
        t1 (float): Hopping parameter for odd-indexed bonds.
        t_scale (np.ndarray): Array of hopping scale factors.

    Returns:
        np.ndarray: Hamiltonian matrix of size (N, N).
    """
    hamiltonian = np.zeros((N, N))

    for i in range(N):
        # Periodic boundary conditions
        if i == N - 1:
            hopping = t1 if i % 2 == 0 else t0
            hamiltonian[i][0] = -hopping * t_scale[i]
            hamiltonian[0][i] = -hopping * t_scale[i]
        else:
            hopping = t0 if i % 2 == 0 else t1
            hamiltonian[i][i + 1] = -hopping * t_scale[i]
            hamiltonian[i + 1][i] = -hopping * t_scale[i]
    return hamiltonian

def compute_band_gap(sorted_energies):
    """
    Compute the band gap from sorted energy levels.

    Parameters:
        sorted_energies (np.ndarray): 2D array of sorted energy levels for different k-points.
                                      Shape is (number of k-points, number of bands).

    Returns:
        np.ndarray: Array of band gaps between the valence and conduction bands.
    """
    N = sorted_energies.shape[1]
    band_gaps = sorted_energies[:, N // 2] - sorted_energies[:, N // 2 - 1]
    return band_gaps
