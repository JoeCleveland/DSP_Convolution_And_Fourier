import numpy as np
import matplotlib.pyplot as plt

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
    axs[0].set_title('Histogram a[n]')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')

    # Plot histogram of b[n]
    axs[1].bar(center_b, pmf_b, width=(bins_b[1] - bins_b[0]), alpha=0.7)
    axs[1].set_title('Histogram b[n]')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Frequency')

    axs[2].bar(center_y, pmf_y, width=(bins_y[1] - bins_y[0]), alpha=0.7)
    axs[2].set_title('Histogram y[n]')
    axs[2].set_xlabel('Value')
    axs[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig("hist_1.svg")

random_sequence()







n_hops = signal_len // hop_size

stft_output = np.zeros((n_hops, window_size // 2))

for n in range(n_hops):
    idx = n * hop_size
    block = signal[idx:idx + window_size]
    print(block.shape, win.shape)
    block = block * win
    block = np.fft.fft(block)
    stft_output[n] = block[:window_size // 2]

return stft_output
