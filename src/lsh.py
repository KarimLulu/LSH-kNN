from collections import defaultdict
from operator import itemgetter

import numpy as np

from settings import INTEGRATION_PRECISON
from src.utils import nnz


def _integration(f, a, b):
    p = INTEGRATION_PRECISON
    area = 0.0
    x = a
    while x < b:
        area += f(x + 0.5 * p) * p
        x += p
    return area, None


try:
    from scipy.integrate import quad as integrate
except ImportError:
    integrate = _integration


def _false_positive_probability(threshold, b, r):
    prob = lambda s: 1 - (1 - s ** float(r)) ** float(b)
    a, err = integrate(prob, 0.0, threshold)
    return a


def _false_negative_probability(threshold, b, r):
    prob = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
    a, err = integrate(prob, threshold, 1.0)
    return a


def _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight):
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHashLSH(object):
    def __init__(
        self,
        num_permutations=128,
        max_candidates_ratio=3,
        threshold=0.9,
        weights=(0.5, 0.5),
        params=None,
    ):
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        if params is not None:
            self.num_tables, self.k = params
            if self.num_tables * self.k > num_permutations:
                raise ValueError(
                    "The product of `num_tables` and `k` must be less than `num_perm`"
                )
        else:
            false_positive_weight, false_negative_weight = weights
            self.num_tables, self.k = _optimal_param(
                threshold,
                num_permutations,
                false_positive_weight,
                false_negative_weight,
            )
        self.hash_tables = [defaultdict(list) for _ in range(self.num_tables)]
        self.hash_ranges = [
            (i * self.k, (i + 1) * self.k) for i in range(self.num_tables)
        ]
        self.keys = dict()
        self.sorted_hash_tables = [[] for _ in range(self.num_tables)]
        self._min_hashes = {}
        self.max_candidates = max_candidates_ratio * self.num_tables
        self.is_similarity = True

    def add(self, key, minhash):
        if len(minhash) < self.k * self.num_tables:
            raise ValueError("The num_permutatins of MinHash out of range")
        if key in self.keys:
            raise ValueError("The given key has already been added")
        self.keys[key] = [
            self._H(minhash.hash_values[start:end]) for start, end in self.hash_ranges
        ]
        for H, hashtable in zip(self.keys[key], self.hash_tables):
            hashtable[H].append(key)
        self._min_hashes[key] = minhash

    def batch_add(self, minhashes: dict):
        for key, minhash in minhashes.items():
            self.add(key, minhash)

    def index(self):
        for i, hashtable in enumerate(self.hash_tables):
            self.sorted_hash_tables[i] = [H for H in hashtable.keys()]
            self.sorted_hash_tables[i].sort()

    def _query(self, minhash, r, b):
        if r > self.k or r <= 0 or b > self.num_tables or b <= 0:
            raise ValueError("parameter is outside the range")
        # Generate prefixes of concatenated hash values
        hps = [
            self._H(minhash.hash_values[start : start + r])
            for start, _ in self.hash_ranges
        ]
        # Set the prefix length for look-ups in the sorted hash values list
        prefix_size = len(hps[0])
        for ht, hp, hashtable in zip(self.sorted_hash_tables, hps, self.hash_tables):
            i = self._binary_search(len(ht), lambda x: ht[x][:prefix_size] >= hp)
            if i < len(ht) and ht[i][:prefix_size] == hp:
                j = i
                while j < len(ht) and ht[j][:prefix_size] == hp:
                    for key in hashtable[ht[j]]:
                        yield key
                    j += 1

    def query(self, minhash, k):
        if k <= 0:
            raise ValueError("k must be positive")
        if len(minhash) < self.k * self.num_tables:
            raise ValueError("The num_perm of MinHash out of range")
        results = []
        r = self.k
        while r > 0:
            for key in self._query(minhash, r, self.num_tables):
                results.append(key)
                if len(results) >= max(k, self.max_candidates):
                    break
            r -= 1
        # Sort according to similarity
        results = list(set(results))
        similarities = [
            (minhash.jaccard(self._min_hashes[key]), key) for key in results
        ]
        similarities.sort(key=itemgetter(0), reverse=self.is_similarity)
        return [item[-1] for item in similarities[:k]]

    def _binary_search(self, n, func):
        i, j = 0, n
        while i < j:
            h = int(i + (j - i) / 2)
            if not func(h):
                i = h + 1
            else:
                j = h
        return i

    def is_empty(self):
        return any(len(t) == 0 for t in self.sorted_hash_tables)

    def _H(self, hs):
        return bytes(hs.byteswap().data)

    def __contains__(self, key):
        return key in self.keys


class CosineLSH(object):
    def __init__(
        self, hash_size=10, input_dim=100, num_tables=10, max_candidates_ratio=3
    ):
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_tables = num_tables
        self.max_candidates = max_candidates_ratio * self.num_tables
        self.hash_tables = [defaultdict(list) for _ in range(self.num_tables)]
        self.is_similarity = True
        self.keys = dict()
        self._init_hyperplanes()

    def _init_hyperplanes(self):
        self._random_hyperplanes = [
            self._generate_random_vectors() for _ in range(self.num_tables)
        ]

    def _generate_random_vectors(self):
        return np.random.randn(self.hash_size, self.input_dim)

    def _hash(self, x, planes):
        projections = np.dot(planes, x)
        _hash = 0
        for projection in projections:
            _hash = _hash << 1
            if projection >= 0:
                _hash |= 1
        return _hash

    def cosine_distance(self, x, y):
        xor = x ^ y
        number_of_ones = nnz(xor)
        return (self.hash_size - number_of_ones) / self.hash_size

    def query(self, point, k):
        if k <= 0:
            raise ValueError("k must be positive.")
        if len(point) != self.input_dim:
            raise ValueError("Dimensions should match.")

        candidates = []
        hashes = []
        for i, table in enumerate(self.hash_tables):
            binary_hash = self._hash(point, self._random_hyperplanes[i])
            hashes.append(binary_hash)
            candidates.extend(table.get(binary_hash, []))
            if len(candidates) >= max(k, self.max_candidates):
                break
        # Sort according to a similarity
        candidates = list(set(candidates))
        similarities = []
        for key in candidates:
            sim = 0
            for k, hash_code in enumerate(hashes):
                sim += (
                    self.cosine_distance(hash_code, self.keys[key][k]) * self.hash_size
                )
            sim /= len(hashes) * self.hash_size
            similarities.append((sim, key))
        similarities.sort(key=itemgetter(0), reverse=self.is_similarity)
        return [item[-1] for item in similarities[:k]]

    def index(self, key, point):
        self.keys[key] = [
            self._hash(point, planes) for planes in self._random_hyperplanes
        ]
        for H, hashtable in zip(self.keys[key], self.hash_tables):
            hashtable[H].append(key)

    def batch_index(self, points: dict):
        for key, point in points.items():
            self.index(key, point)
