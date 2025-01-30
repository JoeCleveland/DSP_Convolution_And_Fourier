import numpy as np
import scipy.io as io
import scipy.signal.windows as windows
import matplotlib.pyplot as plt


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
        block = signal[idx:idx + window_size]
        block = block * win
        block = np.fft.fft(block)
        stft_output[n] = block[:window_size // 2]

    return np.log(stft_output.T ** 2)


sr, signal = io.wavfile.read("TwoNote_DPA_31.wav")

stft_out = STFT(signal, windows.blackman, 2048, 1024)

plt.imshow(stft_out)
plt.show()
