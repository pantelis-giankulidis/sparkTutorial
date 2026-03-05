import numpy as np 
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark import SparkContext
from pyspark.sql.functions import lit

spark = SparkSession.builder.appName("PageRankHITS").getOrCreate()
sc = spark.sparkContext

data = sc.textFile("data/graph-full.txt")

M = data.map(lambda x: x.split()).map(lambda x: (int(x[0]), int(x[1]))).groupByKey().mapValues(list)
N = 100

ranks = M.mapValues(lambda _: 1.0/N)

num_iter, beta = 40, 0.8

for _ in range(num_iter):
    my_contribution = M.join(ranks).flatMap(lambda x: [(dest, x[1][1]/len(x[1][0])) for dest in x[1][0]])
    ranks = my_contribution.reduceByKey(lambda a, b: a + b).mapValues(lambda r: beta*r + (1-beta)/N)

#print(ranks.sortBy(lambda x: x[1], ascending=False).take(5))

#print(ranks.sortBy(lambda x: x[1], ascending=True).take(5))


M_T = M.flatMap(lambda x: [(dest, x[0]) for dest in x[1]]).groupByKey().mapValues(list)

a = M_T.mapValues(lambda _: 1.0)
h = M_T.mapValues(lambda _: 1.0)


M = M.cache()
M_T = M_T.cache()

for _ in range(num_iter):
    a = M_T.join(h).flatMap(lambda x: [(dest, x[1][1]) for dest in x[1][0]]).reduceByKey(lambda a, b: a + b)
    max_a = a.values().max()
    a = a.mapValues(lambda r: r/max_a)
    
    h = M.join(a).flatMap(lambda x: [(dest, x[1][1]) for dest in x[1][0]]).reduceByKey(lambda a, b: a + b)
    max_h = h.values().max()
    h = h.mapValues(lambda r: r/max_h)
    
    

print(a.sortBy(lambda x: x[1], ascending=False).take(5))
print("####")
print(a.sortBy(lambda x: x[1], ascending=True).take(5))
print("---")
print(h.sortBy(lambda x: x[1], ascending=False).take(5))
print("####")
print(h.sortBy(lambda x: x[1], ascending=True).take(5))