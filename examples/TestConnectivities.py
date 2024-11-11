"""Example to compare different connectivity approaches."""

import numpy as np
import matplotlib.pyplot as plt
import delaynet as dnet
from delaynet.preparation import gen_data


def calculate_approaches(num_it, l_coupl, methods):
    """Calculate the different connectivity approaches."""

    n = 100
    all_res = np.zeros((n, 4))
    for c_index in range(n):
        coupling = float(c_index) * 0.02
        all_res[c_index, 0] = coupling

        print(f"Computing {c_index + 1} of {n} couplings")

        for _ in range(num_it):
            ts = gen_data(
                generation_method="fMRI",
                ts_len=10000,
                n_nodes=1,
                downsampling_factor=10,
                time_resolution=0.2,
                coupling_strength=coupling,
                noise_initial_sd=1.0,
                noise_final_sd=0.05,
                rng=None,
            )
            ts[2:, 1] += l_coupl * coupling * ts[:-2, 0]

            for m, method in enumerate(methods):
                p_val, _ = dnet.connectivity(ts[:, 0], ts[:, 1], method)
                all_res[c_index, m + 1] += p_val < 0.01

    all_res[:, 1:] /= float(num_it)

    return all_res


def main(methods):
    """Main function. Calculate and plot the different connectivity approaches."""
    all_res = calculate_approaches(num_it=10, l_coupl=0.5, methods=methods)
    plt.figure()
    for m, method in enumerate(methods):
        plt.plot(all_res[:, 0], all_res[:, m + 1], label=method)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(methods=["RC", "GC", "COP"])
