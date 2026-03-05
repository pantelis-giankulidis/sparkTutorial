import numpy as np
import random
import time
import pdb
import unittest
from matplotlib import pyplot as plt
from PIL import Image


def l1(u, v):
    return np.sum(np.abs(u - v))

def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))
    return f

def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low = 0, 
                                   high = num_dimensions,
                                   size = k)
        thresholds = np.random.randint(low = min_threshold, 
                                   high = max_threshold + 1, 
                                   size = k)

        functions.append(create_function(dimensions, thresholds))
    return functions

def hash_vector(functions, v):
    return np.array([f(v) for f in functions])

def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))

def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
        any(hashed_point == hashed_A[i]), range(len(hashed_A)))

def lsh_setup(A, k = 24, L = 10):
    functions = create_functions(k = k, L = L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)

# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors = 10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)
    
    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]

# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")

# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors):
    search_range = filter(lambda r: r != query_index, range(len(A)))
    distances = map(lambda r: (r, l1(A[r], A[query_index])), search_range)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]
    return [t[0] for t in best_neighbors]

# TODO: Write a function that computes the error measure
def error(z, lshNeighbors, linearNeighbors) -> int:
    res, res2 = 0, 0
    for i in range(len(lshNeighbors)):
        res += l1(z, linearNeighbors[i])
        res2 += l1(z, lshNeighbors[i])
    return res/res2


# TODO: Solve Problem 4
def problem4():
    A = load_data("data/patches.csv")

    L = [10, 12, 14, 16, 18 ,20]
    errors = [0 for _ in range(6)]

    for j in range(len(errors)):
        functions, hashed_A = lsh_setup(A, L = L[j])
        err = 0
        for i in range(0, 1000, 100):
            query_index = i - 1
            num_neighbors = 3
            lsh_neighbors = lsh_search(A, hashed_A, functions, query_index, num_neighbors)
            linear_neighbors = linear_search(A, query_index, num_neighbors)
            err += error(A[i], lsh_neighbors, linear_neighbors)
        errors[j] = err/10
    

    plt.plot(L, errors)
    plt.xlabel("L")
    plt.ylabel("Error")
    plt.title("Error vs L")
    plt.show()


    '''k = [16, 18, 20, 22 , 24]
    errors2 = [0 for _ in range(5)]

    for j in range(len(errors2)):
        functions, hashed_A = lsh_setup(A, k = k[j])
        err = 0
        for i in range(0, 1000, 100):
            query_index = i
            num_neighbors = 3
            lsh_neighbors = lsh_search(A, hashed_A, functions, query_index, num_neighbors)
            linear_neighbors = linear_search(A, query_index, num_neighbors)
            err += error(A[i], lsh_neighbors, linear_neighbors)
        errors2[j] = err/10
    

    plt.plot(k, errors2)
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.title("Error vs k")
    plt.show()'''
    return -1

def problem4d():
    A = load_data("data/patches.csv")

    L, k = 16, 24
    functions, hashed_A = lsh_setup(A, L = L, k = k)
    query_index = 99
    num_neighbors = 10
    lsh_neighbors = lsh_search(A, hashed_A, functions, query_index, num_neighbors)
    linear_neighbors = linear_search(A, query_index, num_neighbors)

    
    lsh_distances = [l1(A[query_index], A[neighbor]) for neighbor in lsh_neighbors]
    linear_distances = [l1(A[query_index], A[neighbor]) for neighbor in linear_neighbors]

    plt.figure()
    plt.plot(range(1, num_neighbors + 1), lsh_distances, label='LSH', marker='o')
    plt.plot(range(1, num_neighbors + 1), linear_distances, label='Linear Search', marker='x')
    plt.xlabel('Rank of Neighbor')
    plt.ylabel('L1 Distance')
    plt.title('Distance of Top Neighbors found by LSH vs Linear Search')
    plt.legend()
    plt.grid(True)
    plt.show()
    
#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

    ### TODO: Write your tests here (they won't be graded, 
    ### but you may find them helpful)


if __name__ == '__main__':
    #unittest.main() ### TODO: Uncomment this to run tests
    #problem4()
    problem4d()
