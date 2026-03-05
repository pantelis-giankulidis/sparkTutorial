import numpy as np

np.random.seed(42)
x = np.random.randint(0, 100, size = (15, 10))

U,S,Vt = np.linalg.svd(x, full_matrices=False)

r = 9
U_r = U[:, :r]
S_r = np.diag(S[:r])
Vt_r = Vt[:r, :]

doc_embed = U_r @ S_r
doc_indices = [1, 9, 14]
print(np.round(doc_embed[doc_indices, :], 2))

zoo_embed = S_r @ Vt_r
zoo_indices = [1, 4, 9]
print(np.round(zoo_embed[:, zoo_indices], 2))
