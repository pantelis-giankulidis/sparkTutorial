import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class CountMinSketch:
    def __init__(self, delta, epsilon, hash_params):
        self.w = int(np.ceil(np.e / epsilon))
        self.d = int(np.ceil(np.log(1/delta)))
        self.count = np.zeros((self.d, self.w), dtype=int)
        self.hash_params = hash_params 
        self.p = 123457

    def hash_func(self, a, b, p, n_buckets, x):
        y = x % p
        hash_val = (a * y + b) % p
        return hash_val % n_buckets

    def update(self, item):
        for j in range(self.d):
            idx = self.hash_func(self.hash_params[j][0], self.hash_params[j][1], self.p, self.w, item)
            self.count[j, idx] += 1

    def estimate(self, item):
        estimates = []
        for j in range(self.d):
            idx = self.hash_func(self.hash_params[j][0], self.hash_params[j][1], self.p, self.w, item)
            estimates.append(self.count[j, idx])
        return min(estimates)

delta = np.exp(-5)
epsilon = np.e * 10**-4
hash_params = [(3, 1561), (17, 277), (38, 394), (61, 13), (78, 246)]
cms = CountMinSketch(delta, epsilon, hash_params)


true_counts = {}
t = 0
with open("data/words_stream_tiny.txt", "r") as f:
    for line in f:
        item = int(line.strip())
        cms.update(item)
        true_counts[item] = true_counts.get(item, 0) + 1
        t += 1

frequencies = []
relative_errors = []

for word, F_i in true_counts.items():
    F_tilde_i = cms.estimate(word)
    relative_error = (F_tilde_i - F_i) / F_i
    
    relative_errors.append(relative_error)
    frequencies.append(F_i / t)

# 4. Plotting
plt.figure(figsize=(10, 6))
plt.scatter(frequencies, relative_errors, alpha=0.5, s=10)
plt.xscale('log')
plt.xlabel('Exact Word Frequency (F[i] / t)')
plt.ylabel('Relative Error (Er[i])')
plt.title('Count-Min Sketch: Relative Error vs. Frequency')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()