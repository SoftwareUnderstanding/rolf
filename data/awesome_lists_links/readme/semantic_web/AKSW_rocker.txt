## ROCKER: A Refinement Operator for Key Discovery ##

[![Build Status](http://ci.aksw.org/jenkins/buildStatus/icon?job=Rocker)](http://ci.aksw.org/jenkins/view/All/job/Rocker/)

### Demo ###

A demo of ROCKER for Data Quality is running at http://rocker.aksw.org. It offers a web interface with accessible APIs. For computations on large datasets, please follow the guide below.

### Run from terminal ###

First, download the [full jar package](https://github.com/AKSW/rocker/releases/download/v1.2.1/rocker-1.2.1-full.jar), which also contains all required dependencies. Datasets are available here:

OAEI Benchmark 2011 (artificial data)

* [OAEI_2011_Restaurant_1.nt.gz](https://bitbucket.org/mommi84/rocker-servlet/downloads/OAEI_2011_Restaurant_1.nt.gz)
* [OAEI_2011_Restaurant_2.nt.gz](https://bitbucket.org/mommi84/rocker-servlet/downloads/OAEI_2011_Restaurant_2.nt.gz)

DBpedia 3.9 (real data)

* [album.nt.gz](https://bitbucket.org/mommi84/rocker-servlet/downloads/album.nt.gz)
* [animal.nt.gz](https://bitbucket.org/mommi84/rocker-servlet/downloads/animal.nt.gz)
* [architecturalStruture.nt.gz](https://bitbucket.org/mommi84/rocker-servlet/downloads/architecturalStruture.nt.gz)
* [artist.nt.gz](https://bitbucket.org/mommi84/rocker-servlet/downloads/artist.nt.gz)
* [careerstation.nt.gz](https://bitbucket.org/mommi84/rocker-servlet/downloads/careerstation.nt.gz)
* [musicalWork.nt.gz](https://bitbucket.org/mommi84/rocker-servlet/downloads/musicalWork.nt.gz)
* [organisationMember.nt.gz](https://bitbucket.org/mommi84/rocker-servlet/downloads/organisationMember.nt.gz)
* [personFunction.nt.gz](https://bitbucket.org/mommi84/rocker-servlet/downloads/personFunction.nt.gz)
* [soccerplayer.nt.gz](https://bitbucket.org/mommi84/rocker-servlet/downloads/soccerplayer.nt.gz)
* [village.nt.gz](https://bitbucket.org/mommi84/rocker-servlet/downloads/village.nt.gz)

To run ROCKER:

```
java -Xmx8g -jar rocker-1.2.1-full.jar <dataset name> <dataset path with protocol> <class name> <find one key> <fast search> <alpha threshold>
```

Example:

```
java -Xmx8g -jar rocker-1.2.1-full.jar "restaurant_1" "file:///home/rocker/OAEI_2011_Restaurant_1.nt" "http://www.okkam.org/ontology_restaurant1.owl#Restaurant" false true 1.0
```

We recommend to run your experiments on a machine with at least 8 GB of RAM.

### Maven

```xml
<repository>
    <id>maven.aksw.internal</id>
    <name>University Leipzig, AKSW Maven2 Repository</name>
    <url>http://maven.aksw.org/archiva/repository/internal</url>
</repository>
...
<dependency>
    <groupId>org.aksw.rocker</groupId>
    <artifactId>rocker</artifactId>
    <version>1.3.1</version>
</dependency>
```

### Java library ###

You may also download the [Java library](https://github.com/AKSW/rocker/releases/download/v1.2.1/rocker-1.2.1.jar) without dependencies.

### Basic usage ###

```java
Rocker r = null;
r = new Rocker("restaurant_1", "file:///home/rocker/OAEI_2011_Restaurant_1.nt",
        "http://www.okkam.org/ontology_restaurant1.owl#Restaurant", false, true, 1.0);
r.run();
Set<CandidateNode> results = r.getKeys();
```

### Citing ROCKER ###

Please refer to the paper *T. Soru, E. Marx, A.-C. Ngonga Ngomo, "ROCKER: A Refinement Operator for Key Discovery"*, in proceedings of the 24th International Conference on World Wide Web, WWW 2015. [[PDF](http://svn.aksw.org/papers/2015/WWW_Rocker/public.pdf)] [[ACM](http://dl.acm.org/citation.cfm?id=2741642)]

```
@inproceedings{Soru:2015:RRO:2736277.2741642,
 author = {Soru, Tommaso and Marx, Edgard and {Ngonga Ngomo}, Axel-Cyrille},
 title = {ROCKER: A Refinement Operator for Key Discovery},
 booktitle = {Proceedings of the 24th International Conference on World Wide Web},
 series = {WWW '15},
 year = {2015},
 isbn = {978-1-4503-3469-3},
 location = {Florence, Italy},
 pages = {1025--1033},
 numpages = {9},
 url = {http://doi.acm.org/10.1145/2736277.2741642},
 doi = {10.1145/2736277.2741642},
 acmid = {2741642},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {key discovery, link discovery, linked data, refinement operators, semantic web},
}
```
