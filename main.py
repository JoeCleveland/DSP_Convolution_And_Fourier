import numpy as np
import matplotlib.pyplot as plt
import scipy


def random_sequence():
    a = np.random.uniform(low=-1, high=1, size=10000)
    b = np.random.uniform(low=-1, high=1, size=10000)
    y = a * b

    hist_a, bins_a = np.histogram(a, 100)
    hist_b, bins_b = np.histogram(b, 100)
    hist_y, bins_y = np.histogram(y, 100)

    center_a = (bins_a[:-1] + bins_a[1:]) * 0.5
    pmf_a = hist_a / np.sum(hist_a)

    center_b = (bins_b[:-1] + bins_b[1:]) * 0.5
    pmf_b = hist_b / np.sum(hist_b)

    center_y = (bins_y[:-1] + bins_y[1:]) * 0.5
    pmf_y = hist_y / np.sum(hist_y)

    # Plot histogram of a[n]
    fig, axs = plt.subplots(1, 3)
    axs[0].bar(center_a, pmf_a, width=(bins_a[1] - bins_a[0]), alpha=0.7)
    axs[0].set_title("Histogram a[n]")
    axs[0].set_xlabel("Value")
    axs[0].set_ylabel("Frequency")

    # Plot histogram of b[n]
    axs[1].bar(center_b, pmf_b, width=(bins_b[1] - bins_b[0]), alpha=0.7)
    axs[1].set_title("Histogram b[n]")
    axs[1].set_xlabel("Value")
    axs[1].set_ylabel("Frequency")

    axs[2].bar(center_y, pmf_y, width=(bins_y[1] - bins_y[0]), alpha=0.7)
    axs[2].set_title("Histogram y[n]")
    axs[2].set_xlabel("Value")
    axs[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("plots/hist_1.png")


# random_sequence()


def convolution():
    N = 101
    # Generate two triangular windows
    a = np.array([1 - abs((i - (N - 1) / 2) / ((N - 1) / 2)) for i in range(N)])
    b = np.array([1 - abs((i - (N - 1) / 2) / ((N - 1) / 2)) for i in range(N)])

    # Perform convolution: we now apply the convolution function to the generated windows.
    L_a = len(a)
    L_b = len(b)
    L_y = L_a + L_b - 1  # Length of the output signal
    y = [0] * L_y

    # convolution step by step
    for n in range(L_y):
        for k in range(L_a):
            if 0 <= n - k < L_b:
                y[n] += a[k] * b[n - k]

    # Plot the results. We visualize:

    # The first triangular window.
    # The second triangular window.
    # The result of their convolution.
    # Plot the first triangular window
    plt.figure(figsize=(6, 4))
    plt.plot(a, label="a[n]", color="blue")
    plt.title("Triangular Window a[n]")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/triangular_window_a.png")

    # Plot the second triangular window
    plt.figure(figsize=(6, 4))
    plt.plot(b, label="b[n]", color="orange")
    plt.title("Triangular Window b[n]")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/triangular_window_b.png")

    # Plot the result of the convolution
    plt.figure(figsize=(6, 4))
    plt.plot(y, label="y[n] = a[n] * b[n]", color="green")
    plt.title("Convolution Result y[n]")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/convolution_result_y.png")

    
    plt.figure(figsize=(12, 8))
    
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
    plt.savefig("plots/joint_convolution_plot.png")

    # return a, b, y


def DFT():
    sample_rate = 16_000
    duration = 2
    signal_params = [(0.5, 1000), (0.25, 2000), (0.25, 2020)]
    file_name = "audio/section_2_DFT.wav"
    timeAxis = np.linspace(0, 2, sample_rate * duration)

    # We have a sum of the frequencies that form the signal x[n]
    list_of_signals = np.array(
        [amp * np.sin(2 * np.pi * freq * timeAxis) for amp, freq in signal_params]
    )

    # we sum up the signals to create one signal
    x = np.sum(list_of_signals, axis=0)

    scipy.io.wavfile.write(file_name, sample_rate, x)
    print(f"file saved at: {file_name}")

    _, data = scipy.io.wavfile.read(file_name)

    # Pad data to the next power of 2
    next_pow2 = int(2 ** np.ceil(np.log2(len(data))))
    padded_data = np.pad(data, (0, next_pow2 - len(data)), mode="constant")

    # perform fft
    X = scipy.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), 1 / sample_rate)[:sample_rate]

    # normalize the magnitude values to reflect the actual amplitude of the signal.
    z = np.abs(X) / len(data)

    z[0] = 1e-10

    # convert to db scale we multiply by 10 because we want the power ratio
    z_db = 10 * np.log10(z)
    z_db = z_db[:sample_rate]  # limit length of signal to sample rate

    # plot the
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, z_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB-scaled)")
    plt.ylim(-50, 0)
    plt.xlim(0, sample_rate // 2)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig("plots/section_2_fequency_plot.png")
    print("saved the plot at: section_2_fequency_plot.png")
    # extract the real and complex part to get the phases
    a = X.real
    b = X.imag

    # phi = arctan(b/a)
    phi = np.arctan2(b, a)
    phi = phi[:sample_rate]

    plt.figure(figsize=(12, 6))
    plt.plot(freqs, phi, label="Phase (radians)")

    # Adding grid
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # Adjust y-axis to show ticks as multiples of π/2
    plt.yticks(
        ticks=np.arange(-np.pi, np.pi + np.pi / 2, np.pi / 2),
        labels=[
            r"$-\pi$",
            r"$-\pi/2$",
            r"$0$",
            r"$\pi/2$",
            r"$\pi$",
        ],  # Corresponding labels
    )
    # plt.xlim(0, len(freqs))
    # Add labels and legend
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.savefig("plots/section_2_phase_plot.png")
    print("saved the plot at: section_2_phase_plot.png")

    # Unwrapped phase
    phi_unwrapped = np.unwrap(phi)

    plt.figure(figsize=(12, 6))
    plt.plot(freqs, phi_unwrapped, label="Unwrapped Phase (radians)")

    # Adding grid
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # Add labels and legend
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Unwrapped Phase (radians)")
    plt.savefig("plots/section_2_unwrapped_phase_plot.png")
    
    print("saved the plot at: section_2_unwrapped_phase_plot.png")


def STFT(input, window_fn, window_size, hop_size):
    win = window_fn(window_size)
    input_len = input.shape[0]
    # [1, 2, 3, ...]

    end_padding = window_size - (input_len % hop_size)

    # [1, 2, 3, ..., 0, 0]
    signal = np.concatenate((input, np.zeros(end_padding)))
    signal_len = signal.shape[0]

    n_hops = signal_len // hop_size - 1  # (signal_len / hop_size) - 1

    # create a placehodler matrix of size [n_hops x (window_size/2)]
    stft_output = np.zeros((n_hops, window_size // 2))

    for n in range(n_hops):
        idx = n * hop_size
        block = signal[idx : idx + window_size]
        block = block * win
        block = np.fft.fft(block)
        stft_output[n] = np.abs(block[: window_size // 2])

    stft_out = np.log(stft_output.T**2)
    plt.figure()
    plt.imshow(stft_out, aspect="auto", origin="lower", cmap="viridis")
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (bins)")
    plt.title("STFT Magnitude")
    plt.tight_layout()
    plt.savefig("plots/stft_output.png")

    print("saved the STFT output")


random_sequence()

convolution()

DFT()

sr, signal = scipy.io.wavfile.read("audio/TwoNote_DPA_31.wav")
STFT(signal, scipy.signal.windows.blackman, 2048, 1024)
