# `RIQ`: RDF Indexing on Quadruples

## Summary

`RIQ` is a new approach for fast processing of SPARQL queries on large
datasets containing RDF quadruples (or quads). 
(These queries are also called named graph queries.)
`RIQ` employs a *decrease-and-conquer*
strategy: Rather than indexing the entire RDF dataset, `RIQ` identifies
groups of similar RDF graphs and indexes each group separately. During
query processing, `RIQ` uses novel filtering index to first identify
candidate groups that may contain matches for the query. On these
candidates, it executes optimized queries using a conventional SPARQL
processor (e.g., Jena TDB) to produce the final results.

## Publications

* Anas Katib, Praveen Rao, Vasil Slavov. ``[A Tool for Efficiently Processing SPARQL Queries on RDF Quads](http://ceur-ws.org/Vol-1963/paper472.pdf)." In the 16th International Semantic Web Conference (ISWC 2017), 4 pages, Austria, Vienna, October 2017. (demo)

* Anas Katib, Vasil Slavov, Praveen Rao. ``[RIQ: Fast Processing of SPARQL Queries on RDF Quadruples](http://dx.doi.org/10.1016/j.websem.2016.03.005)." In the Journal of Web Semantics (JWS), Vol. 37, pages 90-111, March 2016. (Elsevier) 

* Vasil Slavov, Anas Katib, Praveen Rao, Vinutha Nuchimaniyanda. "Fast Processing of SPARQL Queries on RDF Quadruples." The 8th IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA 2015), Verona, NY, May 2015. (poster)

* Vasil Slavov, Anas Katib, Praveen Rao, Srivenu Paturi, Dinesh Barenkala. ``[Fast Processing of SPARQL Queries on RDF Quadruples](http://arxiv.org/pdf/1506.01333v1.pdf)." [*Proceedings of the 17th International Workshop on the Web and Databases*](http://webdb2014.eecs.umich.edu/) (**WebDB 2014**), Snowbird, UT, June 2014.

## Contributors

***Faculty:*** Praveen Rao (PI)

***PhD Students:*** Vasil Slavov, Anas Katib

***MS Students:*** Srivenu Paturi, Dinesh Barenkala, Vinutha Nuchimaniyanda

## Acknowledgments

This work was supported by the National Science Foundation under Grant Nos. 1115871 and 1620023.
