import numpy as np

from .utils import sample_phonon_number, calculate_hopping_scale_factor, construct_hamiltonian, compute_band_gap
from .constants import HBAR, beta, scale_factor, a, m

def generate_perturbed_lattices(num_samples, N, omega, q, T, a=a, m=m):
    """
    Generate perturbed lattices by summing contributions from each phonon mode.

    Parameters:
        num_samples (int): Number of perturbed lattice samples to generate.
        N (int): Number of atoms in the lattice.
        omega (np.ndarray): Array of phonon frequencies.
        q (np.ndarray): Array of wavevectors.
        T (float): Temperature (K) for phonon excitation.
        a (float, optional): Lattice constant. Default is imported from `constants`.
        m (float, optional): Atomic mass. Default is imported from `constants`.

    Yields:
        np.ndarray: Array of atomic displacements for each sample.
    """
    for _ in range(num_samples):

        # Initialize displacement array
        u = np.zeros(N) # displacement array
        A = np.zeros(N) # amplitude array

        # Sum displacement from each phonon mode
        for i in range(1, N):  # mode index, skipping omega = 0 as this only shifts the entire crystal
            
            # Sample phonon number from Bose-Einstein distribution
            n = sample_phonon_number(omega[i], T)
            
            # Avoid division by zero for omega=0
            if omega[i] == 0:
                continue
            
            # Calculate corresponding amplitude
            A[i] = np.sqrt(2 * HBAR / (N * m * omega[i]) * (n + 0.5))
            
            # Random phase factor (U[0, 2pi])
            phase = 2 * np.pi * np.random.rand()
            
            # Apply phase to displacements
            u += A[i] * np.sin(phase + q[i] * np.arange(N) * a)

        yield u

def simulate_energy_levels(num_samples, N, omega, q, T, t0, t1, beta=beta, scale_factor=scale_factor):
    """
    Simulate energy levels by constructing perturbed Hamiltonians for a set of lattices.

    Parameters:
        num_samples (int): Number of perturbed lattice samples.
        N (int): Number of atoms in the lattice.
        omega (np.ndarray): Array of phonon frequencies.
        q (np.ndarray): Array of wavevectors.
        T (float): Temperature (K) for simulation.
        t0 (float): Hopping parameter for even bonds.
        t1 (float): Hopping parameter for odd bonds.
        beta (float, optional): Decay constant for hopping modulation. Default is imported from `constants`.
        scale_factor (float, optional): Scale factor for displacements. Default is imported from `constants`.

    Returns:
        np.ndarray: Array of sorted energy levels for all samples (shape: num_samples x N).

    """

    sorted_energies_T = np.zeros((num_samples, N))

    for sample, u in enumerate(generate_perturbed_lattices(num_samples, N, omega, q, T)):
        # Calculate hopping scale factor
        t_scale = calculate_hopping_scale_factor(u, beta, scale_factor)
        
        # Construct the perturbed Hamiltonian
        hamtiltonian_T = construct_hamiltonian(N, t0, t1, t_scale)

        # Solve system of equations to get eigenvalues (energies)
        energies_T, _ = np.linalg.eigh(hamtiltonian_T)

        # Store energy-levels for each sample
        sorted_energies_T[sample] = np.sort(energies_T)

    return sorted_energies_T

def simulate_band_gaps(temperatures, num_samples, N, omega, q, t0, t1):
    """
    Simulate the average band gap and its standard deviation at different temperatures.

    Parameters:
        temperatures (list or np.ndarray): Array of temperatures (K).
        num_samples (int): Number of samples per temperature.
        N (int): Number of atoms in the lattice.
        omega (np.ndarray): Array of phonon frequencies.
        q (np.ndarray): Array of wavevectors.
        t0 (float): Hopping parameter for even bonds.
        t1 (float): Hopping parameter for odd bonds.

    Returns:
        tuple: Two lists:
            - average_band_gaps: Average band gap at each temperature.
            - std_band_gaps: Standard deviation of the band gap at each temperature.

    """
    average_band_gaps = []
    std_band_gaps = []
    for T in temperatures:
        print(f"Processing Temperature: {T} K")
        band_gaps = []
        
        sorted_energies_T = simulate_energy_levels(num_samples, N, omega, q, T, t0, t1)

        band_gaps = compute_band_gap(sorted_energies_T)

        average_band_gaps.append(np.mean(band_gaps))
        std_band_gaps.append(np.std(band_gaps))

    return average_band_gaps, std_band_gaps


def simulate_displacement(temperatures, num_samples, N, omega, q, histogram_temperature):

    """
    Simulate atomic displacements and compute mean square displacement (MSD) statistics.

    Parameters:
        temperatures (list or np.ndarray): Array of temperatures (K).
        num_samples (int): Number of samples per temperature.
        N (int): Number of atoms in the lattice.
        omega (np.ndarray): Array of phonon frequencies.
        q (np.ndarray): Array of wavevectors.
        histogram_temperature (float): Temperature (K) for which displacement data is stored for histogram analysis.

    Returns:
        tuple: Three elements:
            - average_msd_list (list): Average MSD for each temperature.
            - std_msd_list (list): Standard deviation of MSD for each temperature.
            - displacements_dict (dict): Displacement data for the specified histogram_temperature.
    """
    
    # Lists to store average MSD and standard deviation for each temperature
    average_msd_list = []
    std_msd_list = []

    # Dictionary to store displacement data for each temperature (optional)

    displacements_dict = {}
    for T in temperatures:
        print(f"Processing Temperature: {T} K")
        
        # Initialize variables for MSD calculation
        msd_sum = 0.0
        msd_sum_sq = 0.0
        
        # Initialize list to store displacements of atom 0
        displacements_atom0 = []
        
        for u in generate_perturbed_lattices(num_samples, N, omega, q, T):
            
            # Compute Mean Square Displacement (MSD)
            msd = np.mean(u**2)
            msd_sum += msd
            msd_sum_sq += msd**2
            
            # Store displacement of atom 4
            displacements_atom0.append(u[4])
        
        # Compute average MSD and standard deviation
        average_msd = msd_sum / num_samples
        variance_msd = (msd_sum_sq / num_samples) - (average_msd)**2
        std_msd = np.sqrt(variance_msd) if variance_msd > 0 else 0.0
        
        average_msd_list.append(average_msd)
        std_msd_list.append(std_msd)
        
        # Store displacement data for histogram (e.g., at T=300 K)
        if T == histogram_temperature:
            displacements_dict[T] = np.array(displacements_atom0)
        

    return average_msd_list, std_msd_list, displacements_dict