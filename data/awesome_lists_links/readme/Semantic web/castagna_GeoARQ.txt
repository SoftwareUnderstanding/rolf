GeoARQ 
======

GeoARQ uses Lucene Spatial via an ARQ property function to allow to search
nearby a location. All the RDF instances which use the WGS84 geo positioning 
RDF vocabulary [1] to represent geographic coordinates are indexed.

This is *experimental* (and unsupported).


How to use it
-------------

This is how you build an index from a Jena Model:

    ModelIndexerSubject indexer = new ModelIndexerSubject("target/lucene");
    indexer.indexStatements(model.listStatements());
    indexer.close();

This is how you configure ARQ to use the spatial Lucene index:
        
    IndexSearcher searcher = IndexSearcherFactory.create("target/lucene");
    GeoARQ.setDefaultIndex(searcher);

This is an example of a SPARQL query using the :nearby property function to find airports close to Bristol (i.e. latitude ~ 51.3000, longitude ~ -2.71000): 

    PREFIX : <http://example/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX geoarq: <http://openjena.org/GeoARQ/property#>
    PREFIX asc: <http://airports.dataincubator.org/schema/> .

    SELECT ?label {
        ?s a asc:LargeAirport .
        ?s rdfs:label ?lavel .
        ?s geoarq:nearby (51.3000 -2.71000) .
    }

Or, within a bounded box:

    PREFIX : <http://example/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX geoarq: <http://openjena.org/GeoARQ/property#>
    PREFIX asc: <http://airports.dataincubator.org/schema/> .

    SELECT ?label {
        ?s a asc:LargeAirport .
        ?s rdfs:label ?lavel .
        ?s geoarq:within (51.3727 -2.72909 51.3927 -2.70909) .
    }


Todo
----

 * Add more tests
 * Clean-up the code
 * Add support for http://www.w3.org/2003/01/geo/wgs84_pos#lat_long
 * Add ability to specify radius (i.e. geoarq:nearby (51.3000 -2.71000 20))
 * Add ability to specify number of results within a radius
   (i.e. geoarq:nearby (51.3000 -2.71000 20 10))
 * Investigate Geohash [2]
 * Double check lucene-misc dependency, is it necessary?
 * Add ability to index remote SPARQL endpoints?
 * ...


 [1] http://www.w3.org/2003/01/geo/wgs84_pos#

 [2] http://en.wikipedia.org/wiki/Geohash