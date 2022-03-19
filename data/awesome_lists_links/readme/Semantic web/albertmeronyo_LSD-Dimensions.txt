LSD Dimensions
==============

All dimensions and Data Structure Definitions (DSDs) of Linked
Statistical Data, codes and concept schemes associated to them, and
endpoints using them.

## What is this?

LSD Dimensions is an aggregator of all `qb:DataStructureDefinition`
and `qb:DimensionProperty` (and their associated triples) that can be
currently found in the Linked Data Cloud (read: the SPARQL endpoints
in Datahub.io). Its purpose is to improve the reusability of
statistical dimensions, codes and concept schemes in the Web of Data,
providing an interface for users (future work: to programs) to search
for resources commonly used to describe open statistical datasets. It
also looks for ways of using semantic descriptions of these resources
to enhance comparability of statistical datasets in Linked Data.

## How does it work?

1. Querying the Datahub.io API for a list of endpoints
2. Querying these endpoints for DSDs, dimensions, codes and concept schemes
3. Storing all in a messy MongoDB instance
4. Displaying it using Boostrap Table

## Online instance

See http://lsd-dimensions.org

## Dependencies

- Python 2.7.5
- SPARQL Wrapper
- pymongo
- Bottle
- Bootstrap
- Bootstrap Table
