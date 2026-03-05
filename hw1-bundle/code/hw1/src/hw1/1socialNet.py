from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark import SparkContext
import pandas as pd

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

socialNet = spark.read.text("./data/soc-LiveJournal1Adj.txt")

socialNet = socialNet.select((split(socialNet['value'],'\t')[0]).alias("id"), (split(socialNet['value'], '\t')[1]).alias("connections"))
socialNet = socialNet.select("id", split(socialNet["connections"], ",").alias("connections"))

# Step 1: Generate all indirect connections (A -> C) via direct connections (A -> B -> C) by calculating the Cartesian product of each person's 'B' connections
socialNet = socialNet.select("id", explode(socialNet["connections"]).alias("node_1"), explode(socialNet["connections"]).alias("node_2"))
socialNet = socialNet.filter(socialNet["node_1"] != socialNet["node_2"])

# Step 2: Reformat the results to list each indirect connection pair only once (A <-> C)
socialNet = socialNet.withColumn("person", least(col("node_1"), col("node_2"))).withColumn("friend", greatest(col("node_1"), col("node_2")))
socialNet = socialNet.select("id", "person", "friend").distinct().orderBy("id", "person", "friend")

# Step 3: Count occurrences of each indirect undirected connection pair (A <-> C)
socialNet = socialNet.groupBy("person", "friend").count().sort(desc("count"))

# Step 4: Normalise the table so each row becomes person | friend | count
socialNet = socialNet.select("person", "friend", "count").unionByName(socialNet.select(socialNet.friend.alias("person"), socialNet.person.alias("friend"), "count"))

# Step 5: For each person, select the top K (K=10) recommended indirect connections based on the highest occurrence counts
w = Window.partitionBy("person").orderBy(col("count").desc(), col("friend").asc())
topKFriends = (
     socialNet
       .withColumn("index", row_number().over(w))
       .filter(col("index") <= 12)
       .drop("index")
)
topKArray = (
     topKFriends
       .groupBy("person")
       .agg(
           sort_array(
               collect_list(struct(col("count"), col("friend"))),
               asc=False
           ).alias("topk")
       )
)

topKArray.filter(topKArray["person"] == "11").show(truncate=False)