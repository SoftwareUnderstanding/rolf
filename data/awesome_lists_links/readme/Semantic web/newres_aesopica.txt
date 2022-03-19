# Aesopica

A Clojure library designed to help create Semantic Web, and in particular Linked Data/RDF based applications. 
It allows the user to create Linked Data using idiomatic Clojure datastructures, and translate them to various RDF formats.

## Example Usage


```clojure
(ns example
   (:require [aesopica.core :as aes]
             [aesopica.converter :as conv]))

(def fox-and-stork-edn
  {::aes/context
   {nil "http://www.newresalhaider.com/ontologies/aesop/foxstork/"
    :rdf "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
   ::aes/facts
   #{[:fox :rdf/type :animal]
     [:stork :rdf/type :animal]
     [:fox :gives-invitation :invitation1]
     [:invitation1 :has-invited :stork]
     [:invitation1 :has-food :soup]
     [:invitation1 :serves-using :shallow-plate]
     [:stork :gives-invitation :invitation2]
     [:invitation2 :has-invited :fox]
     [:invitation2 :has-food :crumbled-food]
     [:invitation2 :serves-using :narrow-mouthed-jug]
     [:fox :can-eat-food-served-using :shallow-plate]
     [:fox :can-not-eat-food-served-using :narrow-mouthed-jug]
     [:stork :can-eat-food-served-using :narrow-mouthed-jug]
     [:stork :can-not-eat-food-served-using :shallow-plate]}})
  
(conv/convert-to-turtle fox-and-stork-edn)
```
## Features

### String, Integer, Boolean, Long and Custom Datatypes

```clojure
(def fox-and-stork-literals-edn
  {::aes/context
   {nil "http://www.newresalhaider.com/ontologies/aesop/foxstork/"
    :rdf "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    :foaf "http://xmlns.com/foaf/0.1/"
    :xsd "http://www.w3.org/2001/XMLSchema#"}
   ::aes/facts
   #{[:fox :rdf/type :animal]
     [:fox :foaf/name "vo"]
     [:fox :foaf/age 2]
     [:fox :is-cunning true]
     [:fox :has-weight 6.8]
     [:stork :rdf/type :animal]
     [:stork :foaf/name "ooi"]
     [:stork :foaf/age 13]
     [:stork :is-cunning true]
     [:dinner1 :has-date {::aes/value "2002-05-30T18:00:00" ::aes/type :xsd/dateTime}]}})
```
### Quads/Named Graphs


```clojure
(def fox-and-stork-reif-edn
  {::aes/context
   {nil "http://www.newresalhaider.com/ontologies/aesop/foxstork/"
    :rdf "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
   ::aes/facts
   #{[:fox :rdf/type :animal :dinner1]
     [:stork :rdf/type :animal :dinner1]
     [:fox :gives-invitation :invitation1 :dinner1]
     [:invitation1 :has-invited :stork :dinner1]
     [:invitation1 :has-food :soup :dinner1]
     [:invitation1 :serves-using :shallow-plate :dinner1]
     [:stork :gives-invitation :invitation2 :dinner2]
     [:invitation2 :has-invited :fox :dinner2]
     [:invitation2 :has-food :crumbled-food :dinner2]
     [:invitation2 :serves-using :narrow-mouthed-jug :dinner2]
     [:fox :can-eat-food-served-using :shallow-plate]
     [:fox :can-not-eat-food-served-using :narrow-mouthed-jug]
     [:stork :can-eat-food-served-using :narrow-mouthed-jug]
     [:stork :can-not-eat-food-served-using :shallow-plate]}})
```

### Blank Nodes 


```clojure
(def fox-and-stork-blank-node-edn
  {::aes/context
   {nil "http://www.newresalhaider.com/ontologies/aesop/foxstork/"
    :rdf "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
   ::aes/facts
   #{[:fox :rdf/type :animal]
     [:stork :rdf/type :animal]
     [:fox :gives-invitation 'invitation1]
     ['invitation1 :has-invited :stork]
     ['invitation1 :has-food :soup]
     ['invitation1 :serves-using :shallow-plate]
     [:stork :gives-invitation 'invitation2]
     ['invitation2 :has-invited :fox]
     ['invitation2 :has-food :crumbled-food]
     ['invitation2 :serves-using :narrow-mouthed-jug]
     [:fox :can-eat-food-served-using :shallow-plate]
     [:fox :can-not-eat-food-served-using :narrow-mouthed-jug]
     [:stork :can-eat-food-served-using :narrow-mouthed-jug]
     [:stork :can-not-eat-food-served-using :shallow-plate]}})
```

### Conversion to Common Formats such as Turtle, Trig, N-Quads, JSON-LD

The conversion utilizes the [Apache Jena](https://jena.apache.org/) library for conversion. 
First the Clojure EDN representation needs to be converted to a [Jena DataSetGraph](http://jena.apache.org/documentation/javadoc/arq/org/apache/jena/sparql/core/DatasetGraph.html) (a Jena representation of a set of graphs).
Afterwards the Clojure functions that utilize and wrap Jena's [RDF I/O technology (RIOT)](https://jena.apache.org/documentation/io/) can be called. 

Assuming `fox-and-stork-edn` is a Clojure EDN representation of RDF, and `conv` the shorthand for the `aesopica.converter` namespace, a conversion to Turtle can be written as:

```clojure
(conv/convert-to-turtle fox-and-stork-edn)
```
See the `aesopica.converter` namespace and related tests for more examples. 

Note that certain formats, such as Turtle, are not designed with quads/named graphs in mind.
In cases such as these, a converter to a format that supports quads need to be used (e.g.: TriG, N-Quads) to not lose information.

## Design Decisions and Tutorial

I have been writing a number of articles about the use of Clojure for creating Linked Data, that is interlinked with the creation of this library:

1. [General Introduction](https://www.newresalhaider.com/post/aesopica-1/)
2. [Datatypes](https://www.newresalhaider.com/post/aesopica-2/)
2. [Named Graphs](https://www.newresalhaider.com/post/aesopica-3/)

## License

Copyright © 2018 Newres Al Haider

Distributed under the Eclipse Public License (see the LICENSE file). 
