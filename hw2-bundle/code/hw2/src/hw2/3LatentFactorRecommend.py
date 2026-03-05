import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

def main():
    # Hyperparameters
    k = 20
    lam = 0.1
    n_epochs = 40
    eta = 0.01
    
    # Data path
    data_path = os.path.join(os.path.dirname(__file__), '../../data/ratings.train.txt')
    
    
    error_list = [0 for _ in range(n_epochs)]
    def load_array(path: str) -> np.ndarray:
        try:
            data = pd.read_csv(path, sep='\t', header=None, names=['user_id', 'item_id', 'rating'])
        except FileNotFoundError:
            print(f"Error: dataset not found at {path}")
            sys.exit(1)
        
        user_ids = data['user_id'].unique()
        item_ids = data['item_id'].unique()
    
        user_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
        item_to_index = {iid: idx for idx, iid in enumerate(item_ids)}
    
        data['user_idx'] = data['user_id'].map(user_to_index)
        data['item_idx'] = data['item_id'].map(item_to_index)

        return data[['user_idx', 'item_idx', 'rating']].values, len(user_ids), len(item_ids)
    

    ratings_arr, n_users, n_items = load_array(data_path)
    
    print(f"Number of users: {n_users}")
    print(f"Number of items: {n_items}")


    np.random.seed(42) # For reproducibility
    P = np.random.rand(n_users, k) * np.sqrt(5/k)
    Q = np.random.rand(n_items, k) * np.sqrt(5/k)
    
    print(f"Starting SGD with k={k}, lambda={lam}, eta={eta}, epochs={n_epochs}")
    
    for epoch in range(n_epochs):
        ratings_arr, n_users, n_items = load_array(data_path)
        
        for i in range(ratings_arr.shape[0] - 1):
            # Get random itemi
            random_item = ratings_arr[i]
            
        
            u_idx = int(random_item[0])
            i_idx = int(random_item[1])
            r_ui = random_item[2]
            
            p_u = P[u_idx, :]
            q_i = Q[i_idx, :]
            
            # Error term epsilon_iu = 2 * (r_ui - q_i . p_u)
            prediction = np.dot(q_i, p_u)
            diff = r_ui - prediction
            epsilon_iu = 2 * diff 
              
            grad_q = -epsilon_iu * p_u + 2 * lam * q_i
            grad_p = -epsilon_iu * q_i + 2 * lam * p_u 
            grad_p = np.clip(grad_p, -1e3, 1e3)
            grad_q = np.clip(grad_q, -1e3, 1e3)
            
            # Let's compute updates using current values
            new_q_i = q_i - eta * grad_q
            new_p_u = p_u - eta * grad_p
            
            Q[i_idx, :] = new_q_i
            P[u_idx, :] = new_p_u
            
        
        user_indices = ratings_arr[:, 0].astype(int)
        item_indices = ratings_arr[:, 1].astype(int)
        true_ratings = ratings_arr[:, 2]
        
        pred_ratings = np.sum(P[user_indices] * Q[item_indices], axis=1)
        sse = np.sum((true_ratings - pred_ratings) ** 2)
        
        reg_term = lam * (np.sum(P**2) + np.sum(Q**2))
        
        total_error = sse + reg_term
        
        print(f"Epoch {epoch+1}: E = {total_error:.4f}")
        error_list[epoch] = total_error

    
    plt.plot(range(n_epochs), error_list)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Error vs Epoch")
    plt.show()

if __name__ == "__main__":
    main()
