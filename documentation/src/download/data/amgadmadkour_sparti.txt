# Query-Centric Semantic Partitioning (SPARTI)

* Adaptively partition based on query-workload
* Precompute Bloom join between the most frequent triples joins (MF-TJ) combinations
* Partition related properties based on a greedy algorithm and a cost model
* Current version is implemented to run over <b>Apache Spark</b>