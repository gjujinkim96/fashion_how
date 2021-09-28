from itertools import permutations

import numpy as np
from scipy import stats


def calculate_weighted_kendall_tau(pred, label, num_rnk):
    """
    calcuate Weighted Kendall Tau Correlation
    """
    pred = pred.numpy()
    label = label.numpy()
    rank_lst = np.array(list(permutations(np.arange(num_rnk), num_rnk)))

    total_count = 0
    total_corr = 0
    for p, l in zip(pred, label):
        corr, _ = stats.weightedtau(num_rnk - 1 - rank_lst[l],
                                    num_rnk - 1 - rank_lst[p])
        total_corr += corr
        total_count += 1
    return total_corr / total_count
