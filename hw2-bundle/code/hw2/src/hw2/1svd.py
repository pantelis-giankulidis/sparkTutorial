import numpy as np 
import scipy.linalg as la

M = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
U, s, V = la.svd(M, full_matrices=False)
print(U)
print(s)
print(V.T)

evals, evecs = la.eigh(M.T @ M)
evals.sort()
evecs.sort(axis=1)
print(evals)
print(evecs)



