from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import numpy as np
import time

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

raw_data = sc.textFile("./data/data.txt")
raw_data = raw_data.map(lambda x: [float(y) for y in x.split(' ')])
centroids = sc.textFile("./data/c1.txt")
centroids = centroids.map(lambda x: [float(y) for y in x.split(' ')])
centroids2 = sc.textFile("./data/c2.txt")
centroids2 = centroids2.map(lambda x: [float(y) for y in x.split(' ')])

def euclidian_distance(p1, p2):
    return sum([(a-b)**2 for a, b in zip(p1, p2)])**0.5

def manhattan_distance(p1, p2):
    return sum([abs(a-b) for a, b in zip(p1, p2)])

def closest_centroid(data_point: list, centroids: list, eulidean: bool):
    min_dist = float('inf')
    closest_c = None
    for c in centroids:
        dist = euclidian_distance(data_point, c) if eulidean else manhattan_distance(data_point, c)
        if dist < min_dist:
            min_dist = dist
            closest_c = c
    return centroids.index(closest_c)


def add_tuples(a, b):
    return [x + y for x, y in zip(a, b)]

def cost_function(data_points: list, centroids: list, eulidean: bool):
    return sum([euclidian_distance(p, centroids[closest_centroid(p, centroids, eulidean)])**2 for p in data_points]) if eulidean else sum([manhattan_distance(p, centroids[closest_centroid(p, centroids, eulidean)]) for p in data_points])

MAX_ITER, NUM_CLUSTERS = 20, 10
costsEuclidean = [0]*MAX_ITER
costsManhattan = [0]*MAX_ITER
costsEuclideanRandom = [0]*MAX_ITER
costsManhattanRandom = [0]*MAX_ITER
euclidean = True 


start_time = time.time()

# first iteration with initial centroids random
# second iteration with initial centroids from c2.txt
for j in range(2):
    curr_centroids_manhattan = centroids.collect() if j == 0 else centroids2.collect()
    curr_centroids_euclidean = centroids.collect() if j == 0 else centroids2.collect()
    for i in range(MAX_ITER):
        if j == 0:
            costsEuclideanRandom[i] = cost_function(raw_data.collect(), curr_centroids_euclidean, True)
            costsManhattanRandom[i] = cost_function(raw_data.collect(), curr_centroids_manhattan, False)
        if j == 1:
            costsEuclidean[i] = cost_function(raw_data.collect(), curr_centroids_euclidean, True)
            costsManhattan[i] = cost_function(raw_data.collect(), curr_centroids_manhattan, False)
        # Broadcast centroids to all workers (best practice, though simple closure capture works for small data)
        bc_centroids_euclidean = sc.broadcast(curr_centroids_euclidean)
        bc_centroids_manhattan = sc.broadcast(curr_centroids_manhattan)
    
        # Map: Assign each point to simplest cluster (index) -> (index, (point, 1))
        # We map to (point, 1) so we can sum the points AND count them in one reduce
        pts_with_cluster_euclidean = raw_data.map(
            lambda p: (closest_centroid(p, bc_centroids_euclidean.value, True), (p, 1))
        )
        pts_with_cluster_manhattan = raw_data.map(
            lambda p: (closest_centroid(p, bc_centroids_manhattan.value, False), (p, 1))
        )
    
        # Reduce: Sum points and counts per cluster
        # Result: (index, (summed_vector, count))
        cluster_stats_euclidean = pts_with_cluster_euclidean.reduceByKey(
            lambda a, b: (add_tuples(a[0], b[0]), a[1] + b[1])
        )
        cluster_stats_manhattan = pts_with_cluster_manhattan.reduceByKey(
            lambda a, b: (add_tuples(a[0], b[0]), a[1] + b[1])
        )

        stats_map_euclidean = cluster_stats_euclidean.collect()
        stats_map_manhattan = cluster_stats_manhattan.collect()
        stats_map_euclidean.sort(key=lambda x: x[0])
        stats_map_manhattan.sort(key=lambda x: x[0])
    
        new_centroids_euclidean = [None] * NUM_CLUSTERS
        new_centroids_manhattan = [None] * NUM_CLUSTERS
        for idx, (summed_point, count) in stats_map_euclidean:
            new_center = [x / count for x in summed_point]
            new_centroids_euclidean[idx] = new_center
        for idx, (summed_point, count) in stats_map_manhattan:
            new_center = [x / count for x in summed_point]
            new_centroids_manhattan[idx] = new_center
        # Handle empty clusters if any (though logic usually assumes all centroids have at least one point if initialized from data)
        # If a cluster is empty, keep the old centroid
        for k in range(NUM_CLUSTERS):
            if new_centroids_euclidean[k] is None:
                new_centroids_euclidean[k] = curr_centroids_euclidean[k]
            if new_centroids_manhattan[k] is None:
                new_centroids_manhattan[k] = curr_centroids_manhattan[k]
            
        curr_centroids_euclidean = new_centroids_euclidean
        curr_centroids_manhattan = new_centroids_manhattan

print(f"Time taken: {time.time() - start_time} seconds")
print("costsEuclidean", costsEuclidean)
print("costsEuclideanRandom", costsEuclideanRandom)
print("costsManhattan", costsManhattan)
print("costsManhattanRandom", costsManhattanRandom)

plt.plot(np.log(costsEuclidean))
plt.plot(np.log(costsEuclideanRandom))
plt.plot(np.log(costsManhattan))
plt.plot(np.log(costsManhattanRandom))
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("K-Means Cost Function")
plt.legend(["Euclidean", "Euclidean Random", "Manhattan", "Manhattan Random"])
plt.savefig("kmeans_cost_plot.png")  # Save instead of show
# plt.show()
