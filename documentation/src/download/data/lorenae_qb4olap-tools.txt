QB4OLAP tools

[QB4OLAP] (http://lorenae.github.io/qb4olap/) is a RDFS Vocabulary for Business Intelligence over Linked Data.

In this project we include a set of prototypes that conform a showcase of what can be done using this vocabulary, focusing on exploring and querying.

For querying we propose a high level OLAP language, called QL, which consists on a set of well-known operators: `rollup, drilldown, slice,` and `dice`. Using the cube metadata, also written using QB4OLAP, we automatically generate SPARQL queries to implement sequences of QL operations.


INSTALLATION

1) download [zip] (https://github.com/lorenae/qb4olap-tools/archive/master.zip) file and extract it 

2) install [nodejs] (https://nodejs.org/en/)

3) open console, go to qb4olap-tools directory (obtained in step 1)

4) install npm packages needed by the application  (`npm install` command, which install the packages liste in packages.json)

5) run the application  (`node qb4olap.js`)!!

