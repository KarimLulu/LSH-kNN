from collections import defaultdict


class MinHashLSHkNN(object):
    def __init__(self, num_permutations=128, num_tables=8):
        if num_tables <= 0 or num_permutations <= 0:
            raise ValueError("`num_permutatons` and `num_tables` must be positive")
        if num_tables > num_permutations:
            raise ValueError("`num_tables` cannot be greater than `num_permutations`")
        self.num_tables = num_tables
        self.k = int(num_permutations / num_tables)
        self.hash_tables = [defaultdict(list) for _ in range(self.num_tables)]
        self.hash_ranges = [
            (i * self.k, (i + 1) * self.k) for i in range(self.num_tables)
        ]
        self.keys = dict()
        self.sorted_hash_tables = [[] for _ in range(self.num_tables)]

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
        results = set()
        r = self.k
        while r > 0:
            for key in self._query(minhash, r, self.num_tables):
                results.add(key)
                if len(results) >= k:
                    return list(results)
            r -= 1
        return list(results)

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
