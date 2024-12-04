import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_displacement_histogram(displacements, temperature):
    """
    Plot a histogram of atomic displacements with a Gaussian fit.

    Parameters:
        displacements (np.ndarray): Array of atomic displacements (Å).
        temperature (float): Temperature (K) at which the displacements were measured.

    Returns:
        None: Displays a histogram plot with a Gaussian fit overlaid.

    Functionality:
        - Fits a Gaussian distribution to the displacement data.
        - Plots a histogram of the displacements.
        - Overlays the best-fit Gaussian curve.
        - Annotates the mean (μ) and standard deviation (σ) of the fitted Gaussian on the plot.

    Example Usage:
        displacements = np.random.normal(0, 0.1, 1000)
        plot_displacement_histogram(displacements, temperature=300)
    """
    # Fit a Gaussian to the displacement data
    mu, sigma = norm.fit(displacements)
    
    # Create histogram
    plt.figure(figsize=(6, 6))
    n, bins_hist, patches = plt.hist(displacements, bins=100, density=True, 
                                    color='black', alpha=0.7, edgecolor='black', label='Displacement Data')
    
    # Plot the Gaussian fit
    best_fit = norm.pdf(bins_hist, mu, sigma)
    plt.plot(bins_hist, best_fit, 'r--', linewidth=2, label=f'Gaussian Fit\n$\mu$={mu:.2f} Å\n$\sigma$={sigma:.2f} Å')
    
    # Set labels and title
    plt.xlabel('Displacement of Atom 0 (Å)', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title(f'Displacement Distribution of Atom 0 at {temperature} K', fontsize=16)
    plt.tight_layout()
    plt.show()



def plot_average_displacement(temperatures, average_msd_list, std_msd_list):
    """
    Plot the average mean square displacement (MSD) as a function of temperature with error bars and a linear trendline.

    Parameters:
        temperatures (np.ndarray): Array of temperatures (K).
        average_msd_list (np.ndarray): Array of average mean square displacement (Å²) values corresponding to the temperatures.
        std_msd_list (np.ndarray): Array of standard deviations (Å²) of the MSD values for error bars.

    Returns:
        None: Displays a plot of MSD vs temperature with error bars and a linear fit.

    Functionality:
        - Plots the average MSD as scatter points with error bars for standard deviation.
        - Fits a linear trendline to the MSD data and overlays the trendline.
        - Annotates the trendline equation on the plot.
        - Sets appropriate labels, title, and limits for the plot.

    Example Usage:
        temperatures = np.array([0, 100, 200, 300, 400, 500])
        average_msd_list = np.array([0.01, 0.02, 0.04, 0.07, 0.1, 0.15])
        std_msd_list = np.array([0.002, 0.003, 0.004, 0.005, 0.006, 0.007])
        plot_average_displacement(temperatures, average_msd_list, std_msd_list)
    """
    plt.figure(figsize=(6, 6))
    # Plot MSD with error bars
    plt.errorbar(temperatures, np.array(average_msd_list)*1e20, 
                yerr=np.array(std_msd_list)*1e20, 
                fmt='o-', color='blue', ecolor='lightgray', elinewidth=3, capsize=0, label='MSD Data')

    # Fit a linear trendline
    coefficients = np.polyfit(temperatures, np.array(average_msd_list)*1e20, 1)
    trendline = np.poly1d(coefficients)
    plt.plot(temperatures, trendline(temperatures), 'r--', 
            label=f'Trendline: y={coefficients[0]:.4f}x + {coefficients[1]:.4f} Å²')

    # Annotate the trendline equation on the plot
    plt.text(0.05, 0.95, f'y = {coefficients[0]:.4f}x + {coefficients[1]:.4f} Å²',
            transform=plt.gca().transAxes, fontsize=12, color='red',
            verticalalignment='top')

    plt.xlabel('Temperature (K)', fontsize=14)
    plt.ylabel('Mean Square Displacement (Å²)', fontsize=14)
    plt.title('Average Mean Square Displacement vs Temperature', fontsize=16)
    plt.xticks(temperatures)
    plt.ylim(bottom=0)
    plt.xlim(-50,950)
    plt.tight_layout()
    plt.show()