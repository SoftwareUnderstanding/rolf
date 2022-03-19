# SPARQL parser
The SPARQL parser is a library that helps to parse and investigate a SPARQL query and to build up and generate SPARQL queries.

## Usage
To parse a query you can run:
```
SPARQLQuery parsedQuery = new SPARQLQuery("SELECT * FROM <http://graph1> WHERE { ?s ?p ?o . }");
```
The parsedQuery object will then contain a java object representation of that query. It will have a hashMap with prefix objects, a type , a list of IStatements, a set of unknowns, possibly a graph and the original query.

### Prefix objects
Those are quiet simple, they map a name on a URL.

### Type
The following types are supported
* SPARQLQuery.Type.ASK
* SPARQLQuery.Type.DESCRIBE
* SPARQLQuery.Type.SELECT
* SPARQLQuery.Type.CONSTRUCT
* SPARQLQuery.Type.UPDATE 

### IStatement
The IStatement interface is an abstraction of SPARQL statements. It includes 'select blocks', 'where blocks', 'update blocks', 'parentheses blocks' and 'simple statements'. These blocks support methods to extract the unknowns, inner blocks (for instance a select block contains a parentheses block and that again contains multiple simple statements), the graph on which it operates as well as some functional methods (to change the graph for instance).

### Unknown
This is just a java String that holds the name of the variable in the query without the '?' (ie. ?mu becomes "mu")

### Graph
This is also just a java String.

## Installation
Adding to the pom:
```
<dependency>
  <groupId>com.tenforce.semtech</groupId>
  <artifactId>SPARQL-parser</artifactId>
  <version>0.0.3</version>
</dependency>
```
