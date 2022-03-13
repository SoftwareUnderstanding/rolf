# Mediation toolkit

It's a lightweight toolkit to implement ontological mediation over RDF.
It uses ontology mappings in order to rewrite SPARQL SELECT queries and to generate SPARQL CONSTRUCT queries to import an external data set.
 
API
--------
 
The tool is divided in the following packages:

* [uk.soton.service.dataset](https://github.com/correndo/mediation/tree/master/src/uk/soton/service/dataset) Provides the classes and interfaces necessaries to manages distributed datasets.
* [uk.soton.service.mediation](https://github.com/correndo/mediation/tree/master/src/uk/soton/service/mediation) Provides the classes and interfaces necessaries to mediate RDF documents and SPARQL queries using graph rewriting rules.
* [uk.soton.service.mediation.algebra](https://github.com/correndo/mediation/tree/master/src/uk/soton/service/mediation/algebra) Provides the classes and interfaces necessaries to manipulate SPARQL at the algebra level.
* [uk.soton.service.mediation.algebra.operation](https://github.com/correndo/mediation/tree/master/src/uk/soton/service/mediation/algebra/operation) Provides the implementation of SPARQL XPath functions.
* [uk.soton.service.mediation.edoal](https://github.com/correndo/mediation/tree/master/src/uk/soton/service/mediation/edoal) Provides the classes and interfaces necessaries to interface with the [EDOAL][edoal] ontology alignment format.

[edoal]: http://alignapi.gforge.inria.fr/edoal.html 

The ontology alignments are represented as RDF files and describe rewriting rules that allows to define class mappings:
```

	[]    <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
              <http://ecs.soton.ac.uk/om.owl#Alignment> ;
      <http://ecs.soton.ac.uk/om.owl#hasEntityAlignment>
              [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                        <http://ecs.soton.ac.uk/om.owl#EntityAlignment> ;
                <http://ecs.soton.ac.uk/om.owl#hasRelation>
                        <http://ecs.soton.ac.uk/om.owl#EQ> ;
                <http://ecs.soton.ac.uk/om.owl#lhs>
                        [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                                  <http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#object>
                                  <http://correndo.ecs.soton.ac.uk/ontology/target#Boiler> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>
                                  <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>
                                  _:b1
                        ] ;
                <http://ecs.soton.ac.uk/om.owl#rhs>
                        [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                                  <http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#object>
                                  <http://correndo.ecs.soton.ac.uk/ontology/source#Kettle> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>
                                  <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>
                                  _:b1
                        ]
              ] ;
```          

...property mappings:

```
      <http://ecs.soton.ac.uk/om.owl#hasEntityAlignment>
              [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                        <http://ecs.soton.ac.uk/om.owl#EntityAlignment> ;
                <http://ecs.soton.ac.uk/om.owl#hasRelation>
                        <http://ecs.soton.ac.uk/om.owl#EQ> ;
                <http://ecs.soton.ac.uk/om.owl#lhs>
                        [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                                  <http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#object>
                                  _:b2 ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>
                                  <http://correndo.ecs.soton.ac.uk/ontology/target#boiler> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>
                                  _:b3
                        ] ;
                <http://ecs.soton.ac.uk/om.owl#rhs>
                        [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                                  <http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#object>
                                  _:b2 ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>
                                  <http://correndo.ecs.soton.ac.uk/ontology/source#hasKettle> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>
                                  _:b3
                        ]
              ] ;
```

...and data manipulation:

```
      <http://ecs.soton.ac.uk/om.owl#hasEntityAlignment>
              [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                        <http://ecs.soton.ac.uk/om.owl#EntityAlignment> ;
                <http://ecs.soton.ac.uk/om.owl#hasFunctionalDependency>
                        [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                                  <http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#object>
                                  [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                                            <http://www.w3.org/1999/02/22-rdf-syntax-ns#Seq> ;
                                    <http://www.w3.org/1999/02/22-rdf-syntax-ns#_1>
                                            _:b4 ;
                                    <http://www.w3.org/1999/02/22-rdf-syntax-ns#_2>
                                            273.15
                                  ] ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>
                                  <http://www.w3.org/2005/xpath-functions/sub> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>
                                  _:b5
                        ] ;
                        <http://ecs.soton.ac.uk/om.owl#hasRelation>
                        <http://ecs.soton.ac.uk/om.owl#EQ> ;
                <http://ecs.soton.ac.uk/om.owl#lhs>
                        [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                                  <http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#object>
                                  _:b4 ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>
                                  <http://correndo.ecs.soton.ac.uk/ontology/target#temp> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>
                                  _:b6
                        ] ;
                <http://ecs.soton.ac.uk/om.owl#rhs>
                        [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                                  <http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#object>
                                  _:b5 ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>
                                  <http://correndo.ecs.soton.ac.uk/ontology/source#hasTemperature> ;
                          <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>
                                  _:b6
                        ]
              ] ;
```              
              
 Once loaded an alignment the tool allows to rewrite a SPARQL SELECT query in order to fit a given schema:
 
 [kettle-boiler] original query:
``` 
	PREFIX  rdfs: <http://www.w3.org/2000/01/rdf-schema#>
	PREFIX  source: <http://correndo.ecs.soton.ac.uk/ontology/source#>
	PREFIX  owl:  <http://www.w3.org/2002/07/owl#>
	PREFIX  rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
	SELECT DISTINCT  ?v ?y ?z ?lt
	WHERE
	{ ?v rdf:type source:Person .
	?v source:hasKettle ?y .
	?v source:hasKettle ?l .
	?y source:hasTemperature 10 .
	?l source:hasTemperature ?lt .
	} LIMIT   10
```

[kettle-boiler] translated query:

```
	SELECT DISTINCT  ?v ?y ?z ?lt
	WHERE
	{ ?v   <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>  <http://correndo.ecs.soton.ac.uk/ontology/target#User> ;
	       <http://correndo.ecs.soton.ac.uk/ontology/target#boiler>  ?y ;
	       <http://correndo.ecs.soton.ac.uk/ontology/target#boiler>  ?l ;
	?y   <http://correndo.ecs.soton.ac.uk/ontology/target#temp>  283.15 .
	?l  <http://correndo.ecs.soton.ac.uk/ontology/target#temp>  ?_12 .
	LET (?lt := ( ?_12 - 273.15 ))
	} LIMIT   10
```	
