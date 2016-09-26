"""
Task:
Do fancy shit.
"""

# Import Built-Ins
import logging
from collections import defaultdict

# Import Third-Party
import numpy as np

# Import Homebrew


def sparse_dtw(s, q, res=.5):
    SM = defaultdict(lambda: 0)
    n, m = len(s), len(q)

    def quantize(series):
        return [(series[i] - min(series)) / (max(series) - min(series))
                for i in range(len(series))]

    def euc_distance(x, y):
        return np.abs(x - y) ** 2

    def lower_neighbors(x, y):
        if x == 0 and y == 0:
            return []
        else:
            coord_a = (x-1, y) if x >= 0 and y >= 0 else None
            coord_b = (x-1, y-1) if x >= 0 and y >= 0 else None
            coord_c = (x, y-1)if x >= 0 and y >= 0 else None
            return [neighbor for neighbor in (coord_a, coord_b, coord_c)
                    if neighbor]

    def upper_neighbors(x, y):
        if x == n-1 and y == m-1:
            return []
        else:
            coord_a = (x+1, y) if x < n and y < m else None
            coord_b = (x+1, y+1) if x < n and y < m else None
            coord_c = (x, y+1)if x < n and y < m else None
            return [neighbor for neighbor in (coord_a, coord_b, coord_c)
                    if neighbor]

    def unblock_upper_neighbors(x, y):
        coords = upper_neighbors(x, y)
        for coord in coords:
            try:
                SM[coord] = euc_distance(*coord)
            except IndexError:
                continue

    def populate_warp():
        S = quantize(s)
        Q = quantize(q)

        lower_bound = 0
        upper_bound = res
        while 0 <= lower_bound <= 1 - res / 2:
            idxS = [i for i in range(n) if (lower_bound <= S[i] <= upper_bound)]
            idxQ = [i for i in range(m) if (lower_bound <= Q[i] <= upper_bound)]

            # update bounds
            lower_bound += + res / 2
            upper_bound = lower_bound + res

            # For indices, calculate euclidic distances and add to warp matrix SM
            for i in idxS:
                for j in idxQ:
                    euc_d = euc_distance(s[i], q[j])
                    if euc_d == 0:
                        SM[(i, j)] = -1
                    else:
                        SM[(i, j)] = euc_d


    def calculate_warp_costs():

        done = False
        while not done:
            reset = False
            current_items = list(SM.items())
            for i, element in current_items:
                if element != 0:

                    # Calc cost for lower neighbors
                    lower_n = lower_neighbors(*i)
                    if lower_n:
                        min_cost = min([SM[c] for c in lower_n if SM[c] >= 0])
                    else:
                        min_cost = 0
                    SM[i] += min_cost

                    # check upper neighbors
                    upper_n = upper_neighbors(*i)
                    if not any(SM[c] > 0 for c in upper_n):
                        print(upper_n, i)
                        unblock_upper_neighbors(*i)
                        reset = True
                        break
                    else:
                        continue

            if not reset:
                done = True

    def calculate_warp_path():
        hop = n-1, m-1
        warping_path = [hop]
        while hop != (0, 0):
            lower_n = lower_neighbors(hop)
            lowest = lower_n[0]
            if len(lower_n) > 1:
                for next_n in lower_n[1:]:
                    lowest = next_n if SM[lowest] > SM[lower_n] else lowest
            else:
                pass

            hop = lowest
            warping_path.append(hop)

        return sorted(warping_path), SM[warping_path[-1]]

    populate_warp()
    SM = calculate_warp_costs()
    return calculate_warp_path()

s = [3, 4, 5, 3, 3]
q = [1, 2, 2, 1, 0]

print(sparse_dtw(s, q))

