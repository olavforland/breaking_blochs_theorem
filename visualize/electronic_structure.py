import numpy as np
import matplotlib.pyplot as plt

def plot_electronic_structure(sorted_energies_T, energies_0, q, T, num_samples):
    """
    Plot the electronic band structure and density of states (DOS) for a material at finite and zero temperatures.

    Parameters:
        sorted_energies_T (np.ndarray): 2D array of sorted energy levels at finite temperature T.
                                        Shape: (num_samples, num_bands).
        energies_0 (np.ndarray): 1D array of energy levels at T = 0 K. Shape: (num_bands,).
        q (np.ndarray): Array of wavevectors (1/Å) corresponding to the bands.
        T (float): Temperature (K) for the finite temperature energy levels.
        num_samples (int): Number of Monte Carlo samples used to calculate finite temperature energies.

    Returns:
        None: Displays two plots:
              1. Band structure (energy vs wavevector).
              2. Density of states (DOS vs energy).

    Functionality:
        - Computes the mean and standard deviation of the energy levels at temperature T.
        - Plots the band structure for T = 0 K and T = T with error bars representing the standard deviation.
        - Computes and plots the density of states (DOS) at T = 0 K and T = T with a shaded region for the standard deviation.
        - Customizes labels, legends, and axes for both subplots.

    Example Usage:
        sorted_energies_T = np.random.normal(0, 0.1, (100, 50))
        energies_0 = np.linspace(-1, 1, 50)
        q = np.linspace(0, 2 * np.pi, 50)
        T = 300
        num_samples = 100
        plot_electronic_structure(sorted_energies_T, energies_0, q, T, num_samples)
    """    
    std_E_T = sorted_energies_T.std(axis=0)

    std_E_T_rev = std_E_T[::-1]
    std_E_T_test = np.concatenate((std_E_T_rev, std_E_T))

    # -----------------------------
    # Create a Figure with Two Subplots
    # -----------------------------

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 6), gridspec_kw={'width_ratios': [2, 1]})

    # -----------------------------
    # Plot the Pseudo Band Structure on ax1
    # -----------------------------

    mean_E_T = sorted_energies_T.mean(axis=0)

    q_reversed = -q[::-1]
    q_test = np.concatenate((q_reversed, q))

    E0_reversed = energies_0[::-1]
    E0_test = np.concatenate((E0_reversed, energies_0))

    mean_E_T_rev = mean_E_T[::-1]
    mean_E_T_test = np.concatenate((mean_E_T_rev, mean_E_T))

    # Plot the unperturbed band structure with markers only
    ax1.plot(q_test, E0_test, label='DOS at T=0 K', color='black',
            linestyle='None', marker='o', markersize=0.02)

    # Plot the mean perturbed band structure with markers only
    ax1.plot(q_test, mean_E_T_test, label=f'Average DOS at T={T} K', color='b',
            linestyle='None', marker='s', markersize=0.02)

    ax1.set_ylim(-2, 2)

    # Plot error bars to represent the standard deviation using fill_between
    ax1.fill_between(q_test, mean_E_T_test - std_E_T_test, mean_E_T_test + std_E_T_test,
                    color='b', alpha=0.15, label='Standard Deviation')

    # Set labels for ax1
    ax1.set_xlabel('Wavevector $k$ (1/Å)', fontsize=12)
    ax1.set_ylabel('Energy (eV)', fontsize=12)


    # -----------------------------
    # Compute and Plot Density of States (DOS) on ax2
    # -----------------------------

    # Define the number of bins for the histogram
    num_bins = 100

    # Combine all energies to determine the global energy range
    all_energies = np.concatenate((energies_0, *sorted_energies_T))
    E_min = np.min(all_energies)
    E_max = np.max(all_energies)

    # Create bin edges
    bins = np.linspace(E_min, E_max, num_bins + 1)

    # Calculate bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute DOS for 0 K
    DOS_0, _ = np.histogram(energies_0, bins=bins, density=True)

    # Initialize an array to store DOS for each sample
    DOS_T_samples = np.zeros((num_samples, num_bins))

    # Compute DOS for each sample
    for sample in range(num_samples):
        DOS_T_samples[sample], _ = np.histogram(sorted_energies_T[sample], bins=bins, density=True)

    # Calculate average DOS and standard deviation across samples
    DOS_T_mean = DOS_T_samples.mean(axis=0)
    DOS_T_std = DOS_T_samples.std(axis=0)

    # Plot the average DOS at finite temperature
    ax2.plot(DOS_T_mean, bin_centers, label=f'Average DOS at T={T} K',
            color='b', linestyle='-', linewidth=1)

    # Fill the standard deviation area
    ax2.fill_betweenx(bin_centers, DOS_T_mean - DOS_T_std, DOS_T_mean + DOS_T_std,
                    color='b', alpha=0.15, label='Standard Deviation')

    # Plot the DOS at T=0 K
    ax2.plot(DOS_0, bin_centers, label='DOS at T=0 K',
            color='black', linestyle=':', linewidth=1)

    ax2.set_ylim(-2, 2)

    # Set labels for ax2
    ax2.set_xlabel('Density of States', fontsize=12)
    ax2.set_ylabel('Energy (eV)', fontsize=12)  # Shared y-axis could be considered

    # Set the x-axis limit of DOS plot to start from zero
    ax2.set_xlim(left=0)

    ax1.axhline(0, color = 'r', linestyle = '-', linewidth = 1, alpha=1)
    ax2.axhline(0, color = 'r', linestyle = '-', linewidth = 1, alpha=1)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the combined plots
    plt.show()


