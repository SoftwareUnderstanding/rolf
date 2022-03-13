# About #

SPLENDID provides a federation infrastructure for distributed, linked RDF data sources. SPARQL queries are executed transparently across a set of pre-configured SPARQL endpoints. Automatic source selection and query optimization is based on statistical information provided by VoiD descriptions.

This is an ongoing research project at the Institute for Web Science and Technologies, University of Koblenz, Germany. Currently, the software is offered as a stable alpha version. New features and updates may be added over time.

# Quick Start #

  * check out the source code from svn (eclipse project)
> > `svn checkout http://rdffederator.googlecode.com/svn/trunk/ splendid`
  * compile sources
  * run `SPLENDID.sh` or `SPLENDID.bat` with a repository configuration and a SPARQL query file as parameters, e.g.
> > `./SPLENDID.sh SPLENDID-config.n3 eval/queries/cd/`

# Customizing SPLENDID federation #

  * choose a set of data sources
  * generate voiD statistics for all data sources
    * download data source dump
    * 1st option: run voiD generator shell script on N-Triples file
> > > `$>scripts/generate_void_description.sh dump.nt void.n3`
    * 2nd option: run Java voiD generator on subject-sorted RDF file
> > > `$>scripts/run_voidgen dump_sorted_by_subject`
  * add `fed:member` definition for each data source to the federation configuration (see `SPLENDID-config.n3`)
    * `rep:repositoryType` is always "west:VoidRepository"
    * `fed:voidDescription` requires an URI pointing to the voiD file
    * `void:sparqlEndpoint` is the URI of the source's SPARQL endpoint (overrides the endpoint definition in the voiD file)
  * run SPLENDID with your custom configuration

# Technical Overview #

[Presentation at Consuming Linked Open Data Workshop (ISWC 2011):](http://www.slideshare.net/OlafGoerlitz/splendid-9858478)

