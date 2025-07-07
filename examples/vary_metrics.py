"""Example script to show the direct usage of the detrending and casuality metrics.

Vary the detrending methods and connectivity metrics and see how the results change.
"""

from itertools import product
from sys import argv

from numpy import zeros, save, floating

from delaynet import detrend, connectivity
from delaynet.evaluation import roc_auc_rank_c
from delaynet.preparation import gen_delayed_causal_network


def test_all_metrics(n_nodes: int = 15, l_dens: float = 0.5, ts_len: int = 1000):
    """Test all detrend-metric combinations.

    Saves the results in a file in the Results folder, named after the input parameters.
    "./Results/Res_{n_nodes}_{l_dens}_{ts_len}.npy"

    :param n_nodes: Number of nodes.
    :type n_nodes: int
    :param l_dens: Density of the adjacency matrix.
    :type l_dens: float
    :param ts_len: Length of time series.
    :type ts_len: int
    """

    # Detrending methods and connectivity metrics to use
    detrends = ["Identity", "Delta", "Second Difference", "Z-Score"]
    metrics = ["COP", "GC", "GC_Bi", "GV", "MI", "OS", "RC", "TE"]

    # Testing all combination on the same random data
    am, wm, all_ts = gen_delayed_causal_network(n_nodes, ts_len, l_dens, (0.5, 1.5), 0)

    all_res = {}
    for detrend_method, metric in product(detrends, metrics):
        print(f"Testing {detrend_method} and {metric}.")

        detrended_ts = [
            detrend(all_ts[n1, :], detrend_method, tuple([n1]), check_kwargs=False)
            for n1 in range(n_nodes)
        ]

        # use connectivity function directly
        rec_net = zeros((n_nodes, n_nodes))
        rec_lag = zeros((n_nodes, n_nodes))
        for n1 in range(n_nodes):
            for n2 in range(n_nodes):
                if n1 == n2:
                    continue
                result = connectivity(ts1=detrended_ts[n1], ts2=detrended_ts[n2], metric=metric)
                if isinstance(result, (float, floating)):
                    rec_net[n1, n2] = result
                elif isinstance(result, tuple) and len(result) == 2:
                    rec_net[n1, n2] = result[0]
                    rec_lag[n1, n2] = result[1]

        # `rec_net, rec_lag = connectivity(detrended_ts, metric)`
        # For this the connectivity function needs a mode where instead of being passed
        # two time series, it is passed all time series and the indices at once.
        # TODO: compare the results of the two methods with test data

        _, auc, rank_c = roc_auc_rank_c(am.astype(int), wm, rec_net)

        all_res[(detrend_method, metric)] = (auc, rank_c)
        print((auc, rank_c))

    # Save results
    save(
        f"./Results/Res_{n_nodes}_{l_dens}_{ts_len}",
        all_res,
    )


if __name__ == "__main__":
    #
    test_all_metrics(
        n_nodes=int(argv[1]), l_dens=int(argv[2]) / 100.0, ts_len=int(argv[3])
    )
