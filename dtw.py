"""
Task:
Descripion of script here.
"""

# Import Built-Ins
import logging

# Import Third-Party
import numpy as np

# Import Homebrew

# Init Logging Facilities
log = logging.getLogger(__name__)


def euc_distance(a, b):
    return np.abs(a - b) ** 2


def quantize(a,b):
    """
    S[k][i] - min(S[k]) / max(S[k]) - min(S[k])
    """

    ret_a = [(a[i] - min(a)) / (max(a) - min(a)) for i in range(len(a))]
    ret_b = [(b[i] - min(b)) / (max(b) - min(b)) for i in range(len(b))]

    return ret_a, ret_b


def get_lower_neighbors(xy, SM, index=False):
    x, y = xy
    lower_neighbors = []
    indices = []

    lower_neighbour_coords = [(x - 1, y), (x, y - 1), (x - 1, y - 1)]
    for coord in lower_neighbour_coords:
        if coord[0] >= 0 and coord[1] >= 0:
            try:
                if SM[coord] != 0:
                    lower_neighbors.append(SM[coord])
                    indices.append(coord)
            except IndexError:
                continue
    if index:
        return [(neighbor, i) for neighbor, i in zip(lower_neighbors, indices)]
    return lower_neighbors


def get_upper_neighbors(xy, SM):
    x, y = xy
    n, m = SM.shape
    upper_neighbors = []
    upper_neighbor_coords = [(x + 1, y), (x, y + 1), (x + 1, y + 1)]
    for coord in upper_neighbor_coords:
        if coord[0] < n and coord[1] < m:
            try:
                if SM[coord] != 0:
                    upper_neighbors.append(SM[coord])
            except IndexError:
                continue
    return upper_neighbors


def unblock_upper_neighbors(xy, SM, s, q):
    # unblock neighbors that are within SM
    n, m = SM.shape

    return SM


def populate_warp(s, q, res):
    n = len(s)
    m = len(q)
    S, Q = quantize(s, q)
    SM = np.zeros((n, m))

    lower_bound = 0
    upper_bound = res

    while 0 <= lower_bound <= 1 - res/2:
        # Gather indices that fulfill criteria
        idxS = [i for i in range(n) if (lower_bound <= S[i] <= upper_bound)]
        idxQ = [i for i in range(m) if (lower_bound <= Q[i] <= upper_bound)]

        # update bounds
        lower_bound += + res/2
        upper_bound = lower_bound + res

        # For indices, calculate euclidic distances and add to warp matrix SM
        for i in idxS:
            for j in idxQ:
                euc_d = euc_distance(s[i], q[j])
                if euc_d == 0:
                    SM[i, j] = -1
                else:
                    SM[i, j] = euc_d
    return SM


def calculat_warp_costs(s, q, SM):
    n, m = SM.shape
    for i, element in np.ndenumerate(SM):
        if SM[i] != 0:
            x, y = i

            # check lower neighbors
            lower_neighbors = get_lower_neighbors(i, SM)
            if lower_neighbors:
                min_cost = min(lower_neighbors) if not min(lower_neighbors) == -1 else 0
            else:
                min_cost = 0
            SM[i] = SM[i] + min_cost

            # Check upper neighbors
            upper_neighbors = get_upper_neighbors(i, SM)

            # Check for unblocked upper neighbors
            if len(upper_neighbors) == 0:
                upper_neighbor_coords = [(x + 1, y), (x, y + 1), (x + 1, y + 1)]
                for coord in upper_neighbor_coords:
                    if coord[0] < n and coord[1] < m:
                        try:
                            SM[coord] = euc_distance(s[coord[0]], q[coord[1]])
                        except IndexError:
                            continue
    return SM


def sparse_dtw(s, q, res=.5):
    n = len(s)
    m = len(q)

    #  Create Warp Matrix
    SM = populate_warp(s, q, res)

    #  Calculate Warp costs
    SM = calculat_warp_costs(s, q, SM)

    # Calculate warp path
    warping_path = []
    hop = (n-1, m-1)

    warping_path.append(hop)
    while hop != (0, 0):
        lower_neighbors = get_lower_neighbors(hop, SM, index=True)
        min_cost, next_hop = min(lower_neighbors, key=lambda x: x[0])
        hop = next_hop
        warping_path.append(hop)

    return warping_path





if __name__ == '__main__':
    S = [3, 4, 5, 3, 3]
    Q = [1, 2, 2, 1, 0]
    print(sparse_dtw(S, Q))