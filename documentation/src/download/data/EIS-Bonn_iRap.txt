iRap
=======

Interest-based RDF update propagation framework

iRap is an RDF update propagation framework that propagates only interesting parts of an update from the source to the target dataset. 
iRap filters interesting parts of changesets from the source dataset based on graph-pattern-based interest expressions registered by a target dataset user.
This repository provides source code and evaluation material for iRap framework.

Configuration
=========

Interest expressions should be specified in RDF. Basic components of interest expression are:

- **Subscriber:** is an object that identifies a target dataset and preferences associated with it.

- **Interest:** is an object that identifies an interest expression of a `Subscriber` dataset.

iRap interest expression ontology: `@prefix irap: <http://eis.iai.uni-bonn.de/irap/ontology/>`

## Subscriber

`Subscriber` instance contains the following setting:

* **irap:targetType :** type of dataset as a target to the changes. Valid type are: *TDB, SPARQL_ENDPOINT, and VIRTUOSO_JDBC*

* **irap:targetEndpoint :** path to target dataset. If target type is TDB, then endpoint value is path to TDB folder. If target type is SPARQL_ENDPOINT then irap:targetEndpoint is URI to sparql endpoint (This can be public endpoint for querying only or update enabled endpoint).  VIRTUOSO_JDBC type not supported for this  release.

* **irap:targetUpdateURI :** same as irap:targetEndpoint if target type is TDB and update enabled SPARQL_ENDPOINT. If irap:targetType is query only SPARQL_ENDPOINT, then irap:targetUpdateURI should be a URI to SPARQL Update endpoint.

* **irap:piTrackingMethod :** potentially interesting triples tracking method. Valid methods supported are: LOCAL and LIVE_ON_SOURCE

* **irap:piStorageType :** type of dataset for potentially interesting dataset, if irap:piTrackingMethod is LOCAL

* **irap:piStoreBaseURI :** Path (Endpoint URI) to potentially interesting dataset, if irap:piTrackingMethod is LOCAL and irap:piStorageType is TDB (SPARQL_ENDPOINT, respectively)

## Interest

`Interest` instance contains the following setting:

* **irap:subscriber :** URI of a subscriber for this interest

* **irap:sourceEndpoint :** endpoint to the source dataset (SPARQL_ENDPOINT)

* **irap:changesetPublicationType :** location of changeset publication. Valid values are: REMOTE and LOCAL

* **irap:changesetBaseURI :** URI to changeset publication location

* **irap:lastPublishedFilename :** last publication file name to compare with last downloaded changesets by iRap

* **irap:bgp :** interest basic graph pattern (BGP) expression

* **irap:ogp :** optional graph pattern (OGP) expression  





Executing iRap
=========
In order to execute from source, download the code from the repo
`git clone https://github.com/EIS-Bonn/iRap.git`

- Prepare your interest expression (see `Example interest expression` below).
- If your interest expression contains remote changeset publications (such as DBpedia changesets), edit `lastDownloadDate.dat` and adapt the date according to your target dataset
- Running iRap:
				
		$ git clone https://github.com/EIS-Bonn/iRap.git
		$ cd iRap/irap-core
		$ mvn clean install
		$ mvn exec:java -Dexec.args="<interest-exp>,<run-mode>"

  - `interest-exp` specifies an interest expression RDF file

  - `run-mode` specifies how long the changeset manager should run. 
     * -1 - endless, i.e., run 'forever'
     * 0  - one-time, i.e., run until all changesets available are evaluated (DO NOT wait new updates)

		

Example (DBpedia replica)
=========
The following example shows an interest expression for DBpedia remote changesets.

```
# interest.ttl

@prefix : <http://eis.iai.uni-bonn.de/irap/ontology/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@base <http://eis.iai.uni-bonn.de/irap/ontology/> .

###  http://eis.iai.uni-bonn.de/irap/resource/Soccer
<http://eis.iai.uni-bonn.de/irap/resource/Soccer> rdf:type :Interest ;                                                
                                                  :sourceEndpoint "http://live.dbpedia.org/sparql" ;
                                                  :lastPublishedFilename "lastPublishedFile.txt" ;
                                                  :bgp "?a  a <http://dbpedia.org/ontology/Athlete> .  ?a <http://dbpedia.org/property/goals>  ?goals ." ;
                                                  :ogp "?a <http://xmlns.com/foaf/0.1/homepage>  ?page ." ;
                                                  :changesetBaseURI "http://live.dbpedia.org/changesets/" ;
                                                  :changesetPublicationType "REMOTE" ;
                                                  :subscriber <http://eis.iai.uni-bonn.de/irap/resource/Sport.org> .

###  http://eis.iai.uni-bonn.de/irap/resource/Sport.org
<http://eis.iai.uni-bonn.de/irap/resource/Sport.org> rdf:type :Subscriber;
                                                     :piStoreBaseURI "sports-pi-tdb" ;
                                                     :piStorageType "TDB" ;
                                                     :targetType "TDB" ;
                                                     :targetEndpoint "sports-tdb" ;
                                                     :piTrackingMethod "LOCAL" ;
                                                     :targetUpdateURI "sports-tdb" .

```

Dependencies
=========
  1. Java 7

Contact
=======
[iRap mailing list](https://groups.google.com/forum/#!forum/irap-ld)

## License

The source code is under the terms of the [GNU General Public License, version 2](http://www.gnu.org/licenses/gpl-2.0.html).
