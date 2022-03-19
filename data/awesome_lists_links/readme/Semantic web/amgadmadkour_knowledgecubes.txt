![KNOWLEDGECUBES_LOGO](src/main/resources/logo-svg.png)

## About

A Knowledge Cube, or KC for short, is a semantically-guided data management architecture, where data semantics influences the data management architecture rather than a predefined scheme. KC relies on semantics to define how the data is fetched, organized, stored, optimized, and queried. Knowledge cubes use RDF to store data. This allows knowledge cubes to store Linked Data from the Web of Data. Knowledge cubes envisions breaking down the centralized architecture into multiple specialized cubes, each having its own index and data store.

## Quick Start Guide

#### Create Encode Data

```bash
java -cp uber-knowledgecubes-0.1.0.jar:scala-library-2.11.0.jar edu.purdue.knowledgecubes.DictionaryEncoderCLI -i src/main/resources/datasets/original/sample.nt -o /home/amadkour/kclocal/encoded.nt -l /home/amadkour/kclocal -s space
```
The ```kclocal``` will contain the created dictionaries, the initial data structure used by the store

#### Create Store

```bash
spark-submit --master local[*] --class edu.purdue.knowledgecubes.StoreCLI target/uber-knowledgecubes-0.1.0.jar -i /home/amadkour/kclocal/encoded.nt -l /home/amadkour/kclocal -f 0.01 -t roaring -d /home/amadkour/kcdb
```
The database ```kcdb``` directory contains the actual data and reductions for the input NT file. The following is the directory structure of the local store:

#### Run Query Workload

```bash
spark-submit --master local[*] --class edu.purdue.knowledgecubes.BenchmarkReductionsCLI target/uber-knowledgecubes-0.1.0.jar -l /home/amadkour/kclocal -f 0.01 -t roaring -d /home/amadkour/kcdb -q src/main/resources/queries/original
```
The command generates the workload reductions under the ```kcdb/reductions/join``` directory. The partitions are saved using parquet format. 

#### Local Store Overview

```
$ ls
amadkour@amadkour:~/kclocal$ ls
GEFI  dbinfo.yaml  dictionary  encoded.nt  join-reductions.yaml  
results-20200625115017.txt  tables.yaml
```

* ```GEFI```: directory represents the generalized filters created for the input datasets. 
* ```dbinfo.yaml```: file lists meta-data about the store datasets. 
* ```dictionary```: directory containts the string to id mappings created by the dictionary module. 
* ```join-reductions.yaml```: directory contains metadata about the generated reductions.
* ```results-20200625115017.txt```: is the output file containing the query performance output when running the benchmarking modules. 
* ```tables.yaml```: file lists the meta-data about the tables.

#### Database Directory Overview

```
amadkour@amadkour:~/kcdb$ ls
data  reductions
```
The database contains parquet formatted files that represents the original data and reductions:
* ```data```: contains the original data created based on the input NT files.
* ```reductions```: contains the workload-driven reductions created after running a query workload (e.g. after running the Benchmark CLI tool mentioned below). 

Program such as spark-shell can be used to view the parquet file content:

```bash
scala> var data = spark.read.parquet("/home/amadkour/kcdb/reductions/join/13_TRPO_JOIN_13_TRPS")
data: org.apache.spark.sql.DataFrame = [s: int, p: int ... 1 more field]

scala> data.show()
+---+---+---+
|  s|  p|  o|
+---+---+---+
| 11| 13|  3|
| 11| 13|  4|
| 12| 13|  5|
|  8| 13|  1|
+---+---+---+
```

## WORQ: Workload-Driven RDF Query Processing

KC uses a workload-driven RDF query processing technique, or WORQ for short, for filtering non-matching entries during join evaluation as early as possible to reduce the communication and computation overhead. WORQ generates a reduced sets of triples (or reductions, for short) to represent join pattern(s) of query workloads. WORQ can materialize the reductions on disk or in memory and reuses the reductions that share the same join pattern(s) to answer queries. Furthermore, these reductions are not computed beforehand, but are rather computed in an online fashion. KC also answer complex analytical queries that involve unbound properties. Based on a realization of KC on top of Spark, extensive experimentation demonstrates an order of magnitude enhancement in terms of preprocessing, storage, and query performance compared to the state-of-the-art cloud-based solutions.

## Features

* A spark-based API for SPARQL querying
* Efficient execution of frequent workload join patterns
* Materialze workload join patterns in memory or on disk
* Efficiently answer unbound property queries

## Usage

KC provide spark-based API for issuing RDF related operations. There are three steps necessary for running the system: 

* Dictionary Encoding
* Store Creation
* Querying

#### Dictionary Encoding 

KC requires that the dataset be dictionary encoded. The dictionary encoding allows adding resources (subjects or objects) as integers to the filters. 

```bash
java -cp target/uber-knowledgecubes-0.1.0.jar edu.purdue.knowledgecubes.DictionaryEncoderCLI -i [NT File] -o [Output File] -l [Local Path for the new store] -s space
```

The command generates a dictionary encoded version of the dataset. This encoded NT file is used for creating the store. KC automatically encodes and decodes SPARQL queries and the corresponding results. 

#### Store Creation

KC provide the Store class for creation of an RDF store. The input to the store is a spark session, database path where the RDF dataset will be stored, and a local configuration path.

```scala
import org.apache.spark.sql.SparkSession

import edu.purdue.knowledgecubes.GEFI.GEFIType
import edu.purdue.knowledgecubes.GEFI.join.GEFIJoinCreator
import edu.purdue.knowledgecubes.storage.persistent.Store

val localPath = "/path/to/local/path"
val dbPath = "/path/to/db/path"
val ntPath = "/path/to/rdf/file"

val spark = SparkSession.builder
            .appName(s"KnowledgeCubes Store Creator")
            .getOrCreate()

val store = Store(spark, dbPath, localPath)
store.create(ntPath)
```

#### SPARQL Querying

KC provides a SPARQL query processor that takes as input the spark session, database path of where the RDF dataset was created, local configuration file path, a filter type, and a false postivie rate (if any).

```scala
import org.apache.spark.sql.SparkSession

import edu.purdue.knowledgecubes.queryprocessor.QueryProcessor
import edu.purdue.knowledgecubes.GEFI.GEFIType

val spark = SparkSession.builder
            .appName(s"Knowledge Cubes Query")
            .getOrCreate()

val localPath = "/path/to/local/path"
val dbPath = "/path/to/db/path"
val filterType = GEFIType.ROARING // Roaring bitmap
val falsePositiveRate = 0

val queryProcessor = QueryProcessor(spark, dbPath, localPath, filterType, falsePositiveRate)

val query =
  """
    SELECT ?GivenName ?FamilyName WHERE{
        ?p <http://yago-knowledge.org/resource/hasGivenName> ?GivenName . 
        ?p <http://yago-knowledge.org/resource/hasFamilyName> ?FamilyName . 
        ?p <http://yago-knowledge.org/resource/wasBornIn> ?city . 
        ?p <http://yago-knowledge.org/resource/hasAcademicAdvisor> ?a .
        ?a <http://yago-knowledge.org/resource/wasBornIn> ?city .
    }
  """.stripMargin

// Returns a Spark DataFrame containing the results
val r = queryProcessor.sparql(query)

```

#### Constructing Filters

Additionaly, KC provides an API for creating additional filters. KC provides exact and approximate structures for filtering data. Currently KC supports ```GEFIType.BLOOM```, ```GEFIType.ROARING```, and ```GEFIType.BITSET```.

```scala
import org.apache.spark.sql.SparkSession

import edu.purdue.knowledgecubes.GEFI.GEFIType
import edu.purdue.knowledgecubes.GEFI.join.GEFIJoinCreator
import edu.purdue.knowledgecubes.utils.Timer

val spark = SparkSession.builder
            .appName(s"KnowledgeCubes Filter Creator")
            .getOrCreate()
            
var localPath = "/path/to/db/path"
var dbPath = "/path/to/local/path"
var filterType = GEFIType.ROARING
var fp = 0

val filter = new GEFIJoinCreator(spark, dbPath, localPath)
filter.create(filterType, fp)
```

#### Query Execution Benchmarking

KC provides a set of benchmarking classes

* **BenchmarkFilteringCLI:** For benchmarking the query execution when using filters
* **BenchamrkReductionsCLI:** For benchmarking the query execution when using reductions only

    
## Publications

* Amgad Madkour, Ahmed M. Ali, Walid G. Aref, "WORQ: Workload-driven RDF Query Processing", ISWC 2018 [[Paper](https://amgadmadkour.github.io/files/papers/worq.pdf)][[Slides](https://amgadmadkour.github.io/files/presentations/WORQ-ISWC2018.pdf)]

* Amgad Madkour, Walid G. Aref, Ahmed M. Aly, "SPARTI: Scalable RDF Data Management Using Query-Centric Semantic Partitioning", Semantic Big Data (SBD18) [[Paper](https://amgadmadkour.github.io/files/papers/sparti.pdf)][[Slides](https://amgadmadkour.github.io/files/presentations/SPARTI-SBD2018.pdf)]

* Amgad Madkour, Walid G. Aref, Sunil Prabhakar, Mohamed Ali, Siarhei Bykau, "TrueWeb: A Proposal for Scalable Semantically-Guided Data Management and Truth Finding in Heterogeneous Web Sources", Semantic Big Data (SBD18) [[Paper](https://amgadmadkour.github.io/files/papers/trueweb.pdf)][[Slides](https://amgadmadkour.github.io/files/presentations/TrueWeb-SBD2018.pdf)]

* Amgad Madkour, Walid G. Aref, Saleh Basalamah, “Knowledge Cubes - A Proposal for Scalable and Semantically-Guided Management of Big Data”, IEEE BigData 2013 [[Paper](https://amgadmadkour.github.io/files/papers/bigdata2013.pdf)][[Slides](https://amgadmadkour.github.io/files/presentations/KnowledgeCubes.pdf)]

## Contact

If you have any problems running KC please feel free to send an email. 

* Amgad Madkour <amgad@alumni.purdue.edu>
