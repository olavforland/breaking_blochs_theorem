import numpy as np
import matplotlib.pyplot as plt

def plot_band_gaps(temperatures, average_band_gaps, std_band_gaps):
    """
    Plot the relationship between temperature and band gap with error bars and a quadratic fit.

    Parameters:
        temperatures (np.ndarray): Array of temperatures (K).
        average_band_gaps (np.ndarray): Array of average band gap values (eV) corresponding to the temperatures.
        std_band_gaps (np.ndarray): Array of standard deviations (eV) of the band gap values for error bars.

    Returns:
        None: Displays a plot visualizing the band gap vs temperature and prints the coefficients of the quadratic fit.

    Functionality:
        - Plots the average band gap as scatter points.
        - Fits a quadratic curve to the average band gap data and overlays the trendline.
        - Displays error bars for the standard deviations of the band gap data.
        - Prints the coefficients of the quadratic fit in the console.
        - Sets appropriate labels, title, and limits for the plot.

    Example Usage:
        temperatures = np.array([0, 100, 200, 300, 400, 500])
        average_band_gaps = np.array([1.5, 1.45, 1.4, 1.35, 1.3, 1.25])
        std_band_gaps = np.array([0.05, 0.04, 0.03, 0.03, 0.02, 0.02])
        plot_band_gaps(temperatures, average_band_gaps, std_band_gaps)
    """

    plt.figure(figsize=(6, 6))

    # Plot the average band gap as scatter points
    plt.scatter(
        temperatures, average_band_gaps, color='black', label='Average Band Gap', s=100
    )

    # Perform quadratic fit
    coefficients = np.polyfit(temperatures, average_band_gaps, deg=2)

    # Generate a smooth temperature range for plotting the trendline
    temp_fit = np.linspace(temperatures.min(), temperatures.max(), 500)
    trendline_fit = np.polyval(coefficients, temp_fit)

    # Plot the quadratic trendline over the smooth temperature range
    plt.plot(
        temp_fit, trendline_fit, color='red', linestyle='--',
        linewidth=2, label=f'Quadratic Fit\ny = {coefficients[0]:.4e}x² + {coefficients[1]:.4f}x + {coefficients[2]:.4f}'
    )

    # Print the coefficients of the fit
    print("\nQuadratic Fit Coefficients:")
    print(f"a (x² term): {coefficients[0]:.6e}")
    print(f"b (x term): {coefficients[1]:.6f}")
    print(f"c (constant term): {coefficients[2]:.6f}")

    # Plot error bars for standard deviation
    plt.errorbar(
        temperatures, average_band_gaps, fmt='o', color='black',
        yerr=std_band_gaps, ecolor='lightgray', elinewidth=3, capsize=0, alpha=0.5
    )

    # Set x-axis ticks
    plt.xticks(temperatures)

    # Set labels and title
    plt.xlabel('Temperature (K)', fontsize=14)
    plt.ylabel('Band Gap (eV)', fontsize=14)
    plt.title('Change in Band Gap vs Temperature', fontsize=16)
    plt.xlim(-50,950)

    # Enhance layout
    plt.tight_layout()

    # Display the plot
    plt.show()