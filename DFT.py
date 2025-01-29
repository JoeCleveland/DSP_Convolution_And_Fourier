import numpy as np
import scipy
import matplotlib.pyplot as plt


def create_wav(signal_params, duration, sample_rate, filename):
    timeAxis = np.linspace(0, 2, sample_rate*duration)

    # We have a sum of the frequencies that form the signal x[n]
    list_of_signals = np.array([amp*np.sin(2*np.pi*freq*timeAxis)
                                for amp, freq in signal_params])

    # we sum up the signals to create one signal
    x = np.sum(list_of_signals, axis=0)

    scipy.io.wavfile.write(filename, sample_rate, x)
    print(f"file saved at: {filename}")


def DFT(filename):
    samplerate, data = scipy.io.wavfile.read(filename)
    
    # perform fft
    X = scipy.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), 1 / SAMPLE_RATE)[:SAMPLE_RATE]

    # normalize the magnitude values to reflect the actual amplitude of the signal.
    z = np.abs(X)/len(data)

    # convert to db scale
    z_db = 20*np.log10(z)
    z_db = z_db[:SAMPLE_RATE]  # limit length of signal to sample rate

    # fix dc offset
    z[0] = 1e-10

    # plot the
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, z_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB-scaled)")
    plt.ylim(-100, 0)
    # plt.xlim(0, len(freqs))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig("section_2_fequency_plot.png")
    print("saved the plot at: section_2_fequency_plot.png")
    # extract the real and complex part to get the phases
    a = X.real
    b = X.imag

    # phi = arctan(b/a)
    phi = np.arctan2(b, a)
    phi = phi[:SAMPLE_RATE]

    plt.figure(figsize=(12, 6))
    plt.plot(freqs, phi, label="Phase (radians)")

    # Adding grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Adjust y-axis to show ticks as multiples of π/2
    plt.yticks(
        ticks=np.arange(-np.pi, np.pi + np.pi/2, np.pi/2),
        labels=[
            r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"
        ]  # Corresponding labels
    )
    # plt.xlim(0, len(freqs))
    # Add labels and legend
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.savefig("section_2_phase_plot.png")
    print("saved the plot at: section_2_phase_plot.png")


if __name__ == '__main__':
    SAMPLE_RATE = 16_000
    DURATION = 2
    FILE_NAME = "section_2_DFT.wav"
    SIGNAL_PARAMS = [(0.5, 1000), (0.25, 2000), (0.25, 2020)]

    create_wav(SIGNAL_PARAMS, DURATION, SAMPLE_RATE, FILE_NAME)

    DFT(FILE_NAME)
