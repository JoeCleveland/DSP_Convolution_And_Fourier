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
    plt.bar(center_a, pmf_a, width=(bins_a[1] - bins_a[0]), alpha=0.7)
    plt.show()
    # plt.bar(hist_b)

    plt.savefig("hist_a_b.svg")

    # plt.bar(hist_y)

    # plt.savefig("hist_y.svg")

random_sequence()