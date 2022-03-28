SPARQLKit
=========

An implementation of the SPARQL 1.1 Query and Update language in Objective-C.
---------------

This code implements a full SPARQL 1.1 Query and Update engine in Objective-C.
The design is based on trait/role-based programming, where possible
allowing for natural extensibility and component selection/replacement
(e.g. using the Raptor RDF parser and a triple-store backed by the OS X Address Book).

The code depends on the [GTWSWBase framework](https://github.com/kasei/GTWSWBase).

Plugins
-------

The system provides an extensible plugin architecture for data sources and RDF parsers.
Plugins are automatically loaded from the `Library/Application Support/SPARQLKit/PlugIns` directory.

Some example plugins include:

* [GTWSPARQLProtocolStore](https://github.com/kasei/GTWSPARQLProtocolStore) provides triplestore access to remote remote data using the [SPARQL Protocol](http://www.w3.org/TR/sparql11-protocol/)
* [GTWRedland](https://github.com/kasei/GTWRedland) provides both a [librdf](http://librdf.org) in-memory triplestore and a [Raptor](http://librdf.org/raptor/) RDF parser
* [GTWAddressBookTripleStore](https://github.com/kasei/GTWAddressBookTripleStore) provides access to a users address book contacts
* [GTWApertureTripleStore](https://github.com/kasei/GTWApertureTripleStore) provides access to photo metadata (including geographic and depiction data) from [Aperture](http://www.apple.com/aperture/) libraries
* [GTWAOF](https://github.com/kasei/GTWAOF) provides a persistent, append-only quad store


Example
-------

The `gtwsparql` tool available in this package provides a command line interface to a
full SPARQL 1.1 Query and Update environment.

### Loading Data

```
% gtwsparql
sparql> LOAD <http://dbpedia.org/data/Objective-C.ttl> ;
OK
sparql> SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }
-------------
| # | count | 
-------------
| 1 | 372   | 
-------------
sparql> LOAD <http://dbpedia.org/data/SPARQL.ttl> INTO GRAPH <http://example.org/SPARQL> ;
OK
sparql> SELECT * WHERE { GRAPH ?g {} }
-----------------------------------
| # | g                           | 
-----------------------------------
| 1 | <http://example.org/SPARQL> | 
-----------------------------------
sparql> SELECT DISTINCT ?subject WHERE { GRAPH <http://example.org/SPARQL> { ?subject ?p ?o } } ORDER BY ?subject
-----------------------------------------------------------------------------
|  # | subject                                                              | 
-----------------------------------------------------------------------------
|  1 | <http://dbpedia.org/resource/SPARQL>                                 | 
|  2 | <http://fa.dbpedia.org/resource/اسپارکل>                             | 
|  3 | <http://zh.dbpedia.org/resource/SPARQL>                              | 
|  4 | <http://de.dbpedia.org/resource/SPARQL>                              | 
|  5 | <http://dbpedia.org/resource/SPARQL_Protocol_and_RDF_Query_Language> | 
|  6 | <http://ar.dbpedia.org/resource/سباركل>                              | 
|  7 | <http://ru.dbpedia.org/resource/SPARQL>                              | 
|  8 | <http://dbpedia.org/resource/Sparq>                                  | 
|  9 | <http://lv.dbpedia.org/resource/SPARQL>                              | 
| 10 | <http://vi.dbpedia.org/resource/SPARQL>                              | 
| 11 | <http://nl.dbpedia.org/resource/SPARQL>                              | 
| 12 | <http://uk.dbpedia.org/resource/SPARQL>                              | 
| 13 | <http://ja.dbpedia.org/resource/SPARQL>                              | 
| 14 | <http://dbpedia.org/resource/Sparql>                                 | 
| 15 | <http://it.dbpedia.org/resource/SPARQL>                              | 
| 16 | <http://hu.dbpedia.org/resource/Sparql>                              | 
| 17 | <http://wikidata.dbpedia.org/resource/Q54871>                        | 
| 18 | <http://en.wikipedia.org/wiki/SPARQL>                                | 
| 19 | <http://pl.dbpedia.org/resource/SPARQL>                              | 
| 20 | <http://fr.dbpedia.org/resource/SPARQL>                              | 
| 21 | <http://es.dbpedia.org/resource/SPARQL>                              | 
-----------------------------------------------------------------------------
```

### Namespace Completion

The tool can auto-complete prefix declarations (sourced from [prefix.cc](http://prefix.cc/).
By hitting TAB immediately after a prefix name (including colon), the full prefix IRI
is added to the query string:

`sparql> PREFIX foaf:`**&lt;TAB>**  
`sparql> PREFIX foaf: <http://xmlns.com/foaf/0.1/> `

### Configuring the Data Source

The `gtwsparql` tool can take a string argument to specify the data source configuration.
In its simplest form, this is just the name of a triple- or quad-store plugin.
For example, we can query over Aperture photo metadata loaded into the default graph:

```
% gtwsparql -s GTWApertureTripleStore
sparql> PREFIX dcterms: <http://purl.org/dc/terms/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?place (SAMPLE(?i) AS ?image) WHERE { ?i dcterms:spatial [ foaf:name ?place ] FILTER(REGEX(?place, "Airport")) } GROUP BY ?place ORDER BY ?place
-----------------------------------------------------------------------------------------------------------------------------------------------------
| # | place                              | image                                                                                                    | 
-----------------------------------------------------------------------------------------------------------------------------------------------------
| 1 | "Anchorage Airport"                | <file:///Users/greg/Pictures/Aperture Library.aplibrary/Masters/2013/03/27/20130327-000128/P1040654.RW2> | 
| 2 | "Cape Town Airport"                | <file:///Users/greg/Pictures/Aperture Library.aplibrary/Masters/2013/07/04/20130704-222928/IMG_1931.JPG> | 
| 3 | "Genoa Cristoforo Colombo Airport" | <file:///Users/greg/Pictures/Aperture Library.aplibrary/Masters/2013/07/04/20130704-222928/IMG_1796.JPG> | 
| 4 | "Indira Gandhi Airport"            | <file:///Users/greg/Pictures/Aperture Library.aplibrary/Masters/2013/05/22/20130522-235054/IMG_1752.JPG> | 
| 5 | "Or Tambo Airport"                 | <file:///Users/greg/Pictures/Aperture Library.aplibrary/Masters/2013/07/04/20130704-222928/IMG_1924.JPG> | 
| 6 | "Vadodara Airport"                 | <file:///Users/greg/Pictures/Aperture Library.aplibrary/Masters/2013/05/22/20130522-235054/IMG_1762.JPG> | 
| 7 | "Wellington Airport"               | <file:///Users/greg/Pictures/Aperture Library.aplibrary/Masters/2013/04/10/20130410-213038/P1070371.RW2> | 
-----------------------------------------------------------------------------------------------------------------------------------------------------
```

The `SPKTripleModel` can be used to construct a dataset with multiple triplestores, each available in a separate graph.
We can query over both Aperture photo metadata and address book contacts:

```
% gtwsparql -s '{ "storetype": "SPKTripleModel", "graphs": { "tag:addressbook": { "storetype": "GTWAddressBookTripleStore" }, "tag:aperture": { "storetype": "GTWApertureTripleStore" } } }'
sparql> SELECT * WHERE { GRAPH ?g {} }
-------------------------
| # | g                 | 
-------------------------
| 1 | <tag:addressbook> | 
| 2 | <tag:aperture>    | 
-------------------------
```

This allows us to find the number of photos depicting members of the same family by combining depiction data from Aperture with family name data from the address book (by constructing a query dataset using the `FROM` keyword):

```
sparql> PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?family (COUNT(*) AS ?count) FROM <tag:addressbook> FROM <tag:aperture> WHERE { ?image a foaf:Image ; foaf:depicts [ foaf:familyName ?family ] } GROUP BY ?family ORDER BY ?count
----------------------------------
| # | count | family             | 
----------------------------------
| 1 | 1     | "Kjernsmo"         | 
| 2 | 6     | "Heath"            | 
| 3 | 14    | "Brickley"         | 
| 4 | 25    | "Aastrand Grimnes" | 
| 5 | 104   | "Acton"            | 
| 6 | 116   | "Gillis"           | 
| 7 | 504   | "Crawford"         | 
| 8 | 2029  | "Williams"         | 
----------------------------------
```

### Starting an Endpoint

A SPARQL endpoint can easily be started:

```
sparql> endpoint 8080
Endpoint started on port 8080
```

At this point, `http://localhost:8080/sparql` is a [SPARQL Protocol](http://www.w3.org/TR/sparql11-protocol/) endpoint URL that will respond to queries.
