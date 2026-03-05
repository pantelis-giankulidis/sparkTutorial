from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TestHITS").getOrCreate()
sc = spark.sparkContext

# Small graph: 1->2. 2 is a sink.
data = ["1 2"]
rdd = sc.parallelize(data)

M = rdd.map(lambda x: x.split()).map(lambda x: (int(x[0]), int(x[1]))).groupByKey().mapValues(list)
# Expect M keys: [1]

M_T = M.flatMap(lambda x: [(dest, x[0]) for dest in x[1]]).groupByKey().mapValues(list)
# Expect M_T keys: [2]

h = M.mapValues(lambda _: 1.0)
# Expect h keys: [1]
a = M.mapValues(lambda _: 1.0)
# Expect a keys: [1]

print("Initial M keys:", M.keys().collect())
print("Initial M_T keys:", M_T.keys().collect())
print("Initial h keys:", h.keys().collect())
print("Initial a keys:", a.keys().collect())

# Run one iteration of current logic
my_auth = M_T.join(h).flatMap(lambda x: [(dest, x[1][1]) for dest in x[1][0]])
# M_T key=2. h key=1. Join should be empty.
print("Count of my_auth after join:", my_auth.count())

a_new = my_auth.reduceByKey(lambda a, b: a + b)
print("New a keys:", a_new.keys().collect())

my_hub = M.join(a).flatMap(lambda x: [(dest, x[1][1]) for dest in x[1][0]])
h_new = my_hub.reduceByKey(lambda a, b: a + b)
print("New h keys:", h_new.keys().collect())

spark.stop()
