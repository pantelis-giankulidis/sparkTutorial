from pyspark.sql import SparkSession
from itertools import combinations

spark = SparkSession.builder.appName("AprioriLearning").getOrCreate()
sc = spark.sparkContext

raw_data = sc.textFile("./data/browsing.txt")
transactions = raw_data.map(lambda line: set(line.strip().split()))
num_transactions = transactions.count()
min_support = 100 / num_transactions

frequent_1 = transactions.flatMap(lambda x: x) \
    .map(lambda item: (item, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .filter(lambda x: x[1] / num_transactions >= min_support)

L1 = set(frequent_1.map(lambda x: x[0]).collect())
print(f"Frequent 1-itemsets (Item, Count): {len(L1)}")


L2 = list(combinations(sorted(L1), 2))

def count_candidates(transaction, candidates):
    # Check which candidates exist in this specific transaction
    found = []
    for cand in candidates:
        if set(cand).issubset(transaction):
            found.append((cand, 1))
    return found

frequent_2 = transactions.flatMap(lambda tx: count_candidates(tx, L2)) \
    .reduceByKey(lambda a, b: a + b) \
    .filter(lambda x: x[1] / num_transactions >= min_support)

L2 = frequent_2.collect()
print(f"Frequent 2-itemsets (Itemset, Count): {len(L2)}")

support_dict = dict(frequent_1.collect())
confidence_dict = {}
for itemset, count in L2:
    item_a, item_b = itemset
    conf_a_b = count / support_dict[item_a]
    if conf_a_b >= 0.5: # 50% min confidence
        confidence_dict[(item_a, item_b)] = conf_a_b
    
    conf_b_a = count / support_dict[item_b]
    if conf_b_a >= 0.5:
        confidence_dict[(item_b, item_a)] = conf_b_a

confidence_dict = {k: v for k, v in sorted(confidence_dict.items(), key=lambda item: item[1], reverse=True)}
print("Top 10 Association Rules by Confidence:", list(confidence_dict.items())[:10])

L3 = set()
for itemset, count in L2:
    for item in L1:
        if item in itemset:
            continue
        c = (item, *itemset)
        pair1 = (item, itemset[0])
        pair2 = (item, itemset[1])
        if any(pair1 == p[0] for p in L2) and any(pair2 == p[0] for p in L2):
            L3.add(c)


frequent_3 = transactions.flatMap(lambda tx: count_candidates(tx, L3)) \
    .reduceByKey(lambda a, b: a + b) \
    .filter(lambda x: x[1] / num_transactions >= min_support)
L3 = frequent_3.collect()
print(f"Frequent 3-itemsets (Itemset, Count): {len(L3)}")

confidence_dict_3 = {}
for itemset, count in L3:
    item_a, item_b, item_c = itemset
    conf_a_b_c = count / support_dict[item_a]
    if conf_a_b_c >= 0.3: 
        confidence_dict_3[(item_a, item_b, item_c)] = conf_a_b_c
    
    conf_b_a_c = count / support_dict[item_b]
    if conf_b_a_c >= 0.3:
        confidence_dict_3[(item_b, item_a, item_c)] = conf_b_a_c
    
    conf_c_a_b = count / support_dict[item_c]
    if conf_c_a_b >= 0.3:
        confidence_dict_3[(item_c, item_a, item_b)] = conf_c_a_b

confidence_dict_3 = {k: v for k, v in sorted(confidence_dict_3.items(), key=lambda item: item[1], reverse=True)}
print("Top 10 Association Rules by Confidence:", list(confidence_dict_3.items())[:10])
spark.stop()