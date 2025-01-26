import numpy as np
import matplotlib.pyplot as plt

# Parameters:
N = 101  # Length of triangular windows

# Function to perform convolution
def convolution(a, b):
    # Manual implementation of convolution
    y = np.convolve(a, b, mode='full')
    return y

# Function to generate triangular windows and perform convolution
def triangular_windows_and_convolution():
    # Generate two triangular windows
    a = np.array([1 - abs((i - (N - 1) / 2) / ((N - 1) / 2)) for i in range(N)])
    b = np.array([1 - abs((i - (N - 1) / 2) / ((N - 1) / 2)) for i in range(N)])

    # FORMULA FOR THE REPORT: 
    # w[n] = 1 − |n − (N − 1)/2| / ((N − 1)/2), where n = 0, 1, 2, ..., N − 1

    # Perform convolution
    y = convolution(a, b)

    # FORMULA FOR THE REPORT:
    # Convolution equation: y[n] = sum_{k=0}^{N_a-1} a[k] * b[n-k]
    # Length of the output signal: L_y = N_a + N_b - 1

    # Plot the triangular windows and the result of the convolution
    plt.figure(figsize=(12, 8))

    # Plot the first triangular window
    plt.subplot(3, 1, 1)
    plt.plot(a, label='a[n]', color='blue')
    plt.title('Triangular Window a[n]')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot the second triangular window
    plt.subplot(3, 1, 2)
    plt.plot(b, label='b[n]', color='orange')
    plt.title('Triangular Window b[n]')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot the result of the convolution
    plt.subplot(3, 1, 3)
    plt.plot(y, label='y[n] = a[n] * b[n]', color='green')
    plt.title('Convolution Result y[n]')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return a, b, y

# Call the function to execute the task
triangular_windows_and_convolution()
