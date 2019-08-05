import numpy as np

from settings import MAX_HASH, MERSENNE_PRIME, HASH_RANGE


class MinHash(object):
    def __init__(
        self,
        seed=1,
        num_permutations=100,
        permutations=None,
        hash_function=None,
        hash_values=None,
    ):
        self.seed = seed
        self.hash_function = hash_function
        if num_permutations > HASH_RANGE:
            raise ValueError(
                f"Number of permutations should be less than `{MAX_HASH}``"
            )
        self.num_permutations = num_permutations
        if permutations is not None:
            self.permutations = permutations
        else:
            generator = np.random.RandomState(self.seed)
            # map integers to integers - universal hashing
            self.permutations = generator.randint(
                0, MAX_HASH, size=(2, self.num_permutations), dtype=np.int64
            )
        if hash_values is not None:
            self.hash_values = hash_values
        else:
            self.hash_values = self._init_hash_values()
        if len(self.permutations[0]) != len(self):
            raise ValueError(f"Permutations and hash values are of different size.")

    def _init_hash_values(self):
        return np.ones(self.num_permutations, dtype=np.uint64) * MAX_HASH

    def update(self, b):
        hash_value = self.hash_function(b)
        a, b = self.permutations
        phv = ((a * hash_value + b) % MERSENNE_PRIME) & MAX_HASH
        self.hash_values = np.minimum(phv, self.hash_values)

    def jaccard(self, other):
        """Estimate Jaccard distance."""
        if other.seed != self.seed:
            raise ValueError(
                "Cannot compute Jaccard given MinHash with different seeds"
            )
        if len(self) != len(other):
            raise ValueError(
                "Cannot compute Jaccard given MinHash with different numbers "
                "of permutation functions"
            )
        numerator = np.count_nonzero(self.hash_values == other.hash_values)
        denominator = np.float(len(self))
        return np.float(numerator / denominator)

    def clear(self):
        self.hash_values = self._init_hashvalues(len(self))

    def __len__(self):
        return len(self.hash_values)
