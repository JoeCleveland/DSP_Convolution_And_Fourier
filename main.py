import numpy as np
import scipy
import matplotlib.pyplot as plt

# Parameters:
N = 1000

# Function to generate two independent random sequences and plot their histograms
def random_sequence_with_plot():
    # Generate 2 independent random sequences of length N
    # Values are uniformly distributed between -1 and 1
    a = np.random.uniform(-1, 1, N)
    b = np.random.uniform(-1, 1, N)

    # Plot histogram of the 1st random sequence (a[n])
    plt.subplot(1, 2, 1)
    plt.hist(a, bins=100, color='blue', alpha=0.7, label='a[n]')
    plt.title('Histogram a[n]')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot histogram of the 2nd random sequence (b[n])
    plt.subplot(1, 2, 2)
    plt.hist(b, bins=100, color='orange', alpha=0.7, label='b[n]')
    plt.title('Histogram b[n]')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

    return a, b

# Call the function to generate sequences and plot
random_sequence_with_plot()
