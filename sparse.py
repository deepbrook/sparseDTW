"""
Task:
Do fancy shit.
"""

# Import Built-Ins
import logging
from collections import defaultdict

# Import Third-Party
import numpy as np
from scipy.sparse import lil_matrix
# Import Homebrew

class SparseDTW:

    def __init__(self, s, q, res=.5):
        #self.SM = defaultdict(lambda: 0)

        self.n, self.m = len(s), len(q)
        self.res = res
        self.s = s
        self.q = q
        self.SM = lil_matrix((self.n, self.m), dtype=np.int16)

    def quantize(self, series):
        return [(series[i] - min(series)) / (max(series) - min(series))
                for i in range(len(series))]

    def euc_distance(self, x, y):
        return np.abs(x - y) ** 2

    def lower_neighbors(self, x, y):
        if x == 0 and y == 0:
            return []
        else:
            coord_a = (x-1, y) if x-1 >= 0 and y >= 0 else None
            coord_b = (x-1, y-1) if x-1 >= 0 and y-1 >= 0 else None
            coord_c = (x, y-1)if x >= 0 and y-1 >= 0 else None
            return [neighbor for neighbor in (coord_a, coord_b, coord_c)
                    if neighbor is not None]

    def upper_neighbors(self, x, y):
        if x == self.n-1 and y == self.m-1:
            return []
        else:
            coord_a = (x+1, y) if x+1 < self.n and y < self.m else None
            coord_b = (x+1, y+1) if x+1 < self.n and y+1 < self.m else None
            coord_c = (x, y+1) if x < self.n and y+1 < self.m else None
            return [neighbor for neighbor in (coord_a, coord_b, coord_c)
                    if neighbor is not None]

    def unblock_upper_neighbors(self, x, y):
        coords = self.upper_neighbors(x, y)
        if coords:
            for coord in coords:
                try:
                    self.SM[coord] = self.euc_distance(*coord)
                except IndexError:
                    continue

    def populate_warp(self):
        S = self.quantize(self.s)
        Q = self.quantize(self.q)

        lower_bound = 0
        upper_bound = self.res
        while 0 <= lower_bound <= 1 - self.res / 2:
            idxS = [i for i in range(self.n) if (lower_bound <= S[i] <= upper_bound)]
            idxQ = [i for i in range(self.m) if (lower_bound <= Q[i] <= upper_bound)]

            # update bounds
            lower_bound += + self.res / 2
            upper_bound = lower_bound + self.res

            # For indices, calculate euclidic distances and add to warp matrix SM
            for i in idxS:
                for j in idxQ:
                    euc_d = self.euc_distance(self.s[i], self.q[j])
                    if euc_d == 0:
                        self.SM[(i, j)] = -1
                    else:
                        self.SM[(i, j)] = euc_d

    def calculate_warp_costs(self):
        for i in range(self.n):
            for j in range(self.m):
                if self.SM[i,j]:
                    lower_n = self.lower_neighbors(i, j)

                    if lower_n:
                        min_cost = min([self.SM[c] for c in lower_n if self.SM[c] >= 0])
                        min_cost = 0 if min_cost == -1 else min_cost
                    else:
                        min_cost = 0
                    self.SM[i, j] += min_cost

                    # check upper neighbors
                    upper_n = self.upper_neighbors(i, j)
                    if upper_n and not any(self.SM[c] > 0 for c in upper_n):
                        self.unblock_upper_neighbors(i, j)
                    else:
                        continue

    def calculate_warp_path(self):
        hop = self.n-1, self.m-1
        warping_path = [hop]
        while hop != (0, 0):

            lower_n = self.lower_neighbors(*hop)
            lowest = lower_n[0] if lower_n else (0, 0)
            if len(lower_n) > 1:
                for next_n in lower_n[1:]:
                    lowest = next_n if self.SM[lowest] > self.SM[next_n] else lowest
            else:
                pass

            hop = lowest
            warping_path.append(hop)

        return sorted(warping_path), self.SM[warping_path[-1]]

    def __call__(self, *args, **kwargs):
        self.populate_warp()
        self.calculate_warp_costs()

        return self.calculate_warp_path()

    def as_arr(self):
        return self.SM.toarray()


if __name__ == '__main__':
    s = [3, 4, 5, 3, 3]
    q = [1, 2, 2, 1, 0]

    dtw = SparseDTW(s, q)
    print("--------")
    print(dtw.as_arr())
    print("--------")



