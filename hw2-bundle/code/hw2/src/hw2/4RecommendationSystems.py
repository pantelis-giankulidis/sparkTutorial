import numpy as np 
import pandas as pd 
import os 

data_path = os.path.join(os.path.dirname(__file__), '../../data/user-shows.txt')

utility_matrix = []
with open(data_path, 'r') as f:
    for line in f:
        utility_matrix.append(line.strip().split())

utility_matrix = np.array(utility_matrix, dtype=np.float32)

p = np.diag(np.sum(utility_matrix, axis = 1))
q = np.diag(np.sum(utility_matrix, axis = 0))

T = np.diag(1 / np.sqrt(np.diagonal(p)))
user_user_cf = T @ utility_matrix @ utility_matrix.T @ T.T @ utility_matrix

T2 = np.diag(1 / np.sqrt(np.diagonal(q)))
item_item_cf = utility_matrix @ T2 @ utility_matrix.T @ utility_matrix @ T2


# 1
popular_show = np.argpartition(user_user_cf[499, :100], -5)[-5:]
r = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/shows.txt'), sep='\t', header=None)
print(r.iloc[popular_show])

# 2
pop_shows = np.argpartition(item_item_cf[499, :100], -5)[-5:]
print(r.iloc[pop_shows])

