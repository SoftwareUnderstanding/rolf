# JTriple

JTriple is a Java tool which creates a RDF data model out of a Java object model by making use of reflection, a small set of annotations and Jena's flexible RDF/OWL API.

### Why another RDF binding for Java?

The most popular tool for persisting Java objects to RDF is [JenaBean]. JTriple was developed, respectively JenaBean was not modified due to the following reasons:

* JenaBean aims for a persistence layer (object serialization). This fact is often expressed by missing confguration, for instance a field cannot be declared as transient.

* Not the whole functionality of JenaBean is required. Additional data is serialized, for instance the serialization of the package names. Package names are vital for deserialization but for the pure data translation (one-way) it only interferes.

* Data (RDF) and schema (OWL) should be translated into two separate RDF graphs. JenaBean creates only one graph.

## Getting Started

JTriple can be deployed through Maven. Before, the following repository has to be added to your pom.xml

```xml
<repository>
     <id>berlin.reiche.jtriple</id>
     <url>https://github.com/platzhirsch/jtriple/raw/master/repository/releases</url>
</repository>
```

Then it can be added with this dependency

```xml
<dependency>
     <groupId>berlin.reiche.jtriple</groupId>
     <artifactId>jtriple</artifactId>
     <version>0.1-RELEASE</version>
     <scope>compile</scope>
</dependency>
```

Not using Maven? You can also get the [JAR] directly.

### Example

Considering the following example. A class Philosopher

```java
public class Philosopher {

	@RdfIdentifier
	String name;

	String nationality;
	List<Branch> interests;
}
```

with an enum type Branch

```java
public enum Branch {

	EPISTEMOLOGY("Epistemology"),
	MATHEMATIC("Mathematic"),
	METAPHYSISC("Metaphysic"),
	PHILOSOPHY_OF_MIND("Philosophy of Mind");
	
	String name;
	
	Branch(String name) {
		this.name = name;
	}
}
```
The only requirement is to annotate one field or method of a class with `@RdfIdentifier`. Binding objects to RDF is as easy as follows


```java
// create data
Philosopher locke = new Philosopher();
locke.setName("John Locke");
locke.setNationality("English");

List<Branch> branches = new ArrayList<>();
branches.add(METAPHYSISC);
branches.add(EPISTEMOLOGY);
branches.add(PHILOSOPHY_OF_MIND);
locke.setInterests(branches);

// bind object
Binding binding = new Binding(DEFAULT_NAMESPACE);
Model model = binding.getModel();
model.setNsPrefix("philosophy", NAMESPACE);

binding.bind(locke);

// output RDF
model.write(System.out, "TURTLE");
```

It is sufficient to produce this RDF

```
@prefix philosophy:  <http://konrad-reiche.com/philosophy/> .

<http://konrad-reiche.com/philosophy/philosopher/John_locke>
      a       <http://dbpedia.org/page/Philosopher> ;
      philosophy:interests
              <http://konrad-reiche.com/philosophy/branch/Metaphysisc> ,
              <http://konrad-reiche.com/philosophy/branch/Philosophy_of_mind> ,
              <http://konrad-reiche.com/philosophy/branch/Epistemology> ;
      philosophy:name "John Locke"^^<http://www.w3.org/2001/XMLSchema#string> ;
      philosophy:nationality
              "English"^^<http://www.w3.org/2001/XMLSchema#string> .

<http://konrad-reiche.com/philosophy/branch/Epistemology>
      a       philosophy:branch ;
      philosophy:name "Epistemology"^^<http://www.w3.org/2001/XMLSchema#string> .

<http://konrad-reiche.com/philosophy/branch/Metaphysisc>
      a       philosophy:branch ;
      philosophy:name "Metaphysic"^^<http://www.w3.org/2001/XMLSchema#string> .

<http://konrad-reiche.com/philosophy/branch/Philosophy_of_mind>
      a       philosophy:branch ;
      philosophy:name "Philosophy of Mind"^^<http://www.w3.org/2001/XMLSchema#string> .
```

Now, to get more sophisticated results, annotations help to provide neccessary information

```java
@RdfType("http://dbpedia.org/page/Philosopher")
public class Philosopher {

	@Label
	@RdfIdentifier
	String name;

	@RdfProperty("http://www.foafrealm.org/xfoaf/0.1/nationality")
	String nationality;

	List<Branch> interests;
}
```

```java
public enum Branch {

	@SameAs({ "http://dbpedia.org/resource/Epistemology" })
	EPISTEMOLOGY("Epistemology"),
	
	@SameAs({ "http://dbpedia.org/resource/Mathematic" })
	MATHEMATIC("Mathematic"),

	@SameAs({ "http://dbpedia.org/resource/Metaphysic" })
	METAPHYSISC("Metaphysic"),

	@SameAs({ "http://dbpedia.org/resource/Philosophy_of_mind" })
	PHILOSOPHY_OF_MIND("Philosophy of Mind");
	
	@Label
	String name;
	
	Branch(String name) {
		this.name = name;
	}
}
```

Leading to this RDF:

```
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xfoaf:   <http://www.foafrealm.org/xfoaf/0.1/> .
@prefix philosophy:  <http://konrad-reiche.com/philosophy/> .
@prefix dbpedia:  <http://dbpedia.org/resource/> .

<http://konrad-reiche.com/philosophy/philosopher/John_locke>
      a       <http://dbpedia.org/page/Philosopher> ;
      rdfs:label "John Locke"^^<http://www.w3.org/2001/XMLSchema#string> ;
      philosophy:interests
              <http://konrad-reiche.com/philosophy/branch/Metaphysisc> ,
              <http://konrad-reiche.com/philosophy/branch/Philosophy_of_mind> ,
              <http://konrad-reiche.com/philosophy/branch/Epistemology> ;
      xfoaf:nationality "English"^^<http://www.w3.org/2001/XMLSchema#string> .

<http://konrad-reiche.com/philosophy/branch/Metaphysisc>
      a       philosophy:branch ;
      rdfs:label "Metaphysic"^^<http://www.w3.org/2001/XMLSchema#string> ;
      <http://www.w3.org/2002/07/owl#sameAs>
              dbpedia:Metaphysic .

<http://konrad-reiche.com/philosophy/branch/Philosophy_of_mind>
      a       philosophy:branch ;
      rdfs:label "Philosophy of Mind"^^<http://www.w3.org/2001/XMLSchema#string> ;
      <http://www.w3.org/2002/07/owl#sameAs>
              dbpedia:Philosophy_of_mind .

<http://konrad-reiche.com/philosophy/branch/Epistemology>
      a       philosophy:branch ;
      rdfs:label "Epistemology"^^<http://www.w3.org/2001/XMLSchema#string> ;
      <http://www.w3.org/2002/07/owl#sameAs>
              dbpedia:Epistemology .

```

### Annotations

What annotations are there and how can they be used?

<table>
  <tr>
    <th>Name</th><th>Use</th><th>Effect</th>
  </tr>
  <tr>
    <td><code>@RdfIdentifier</code></td><td>Fields, Methods</td><td>Value to be used for constructing the resource URI</td>
  </tr>
  <tr>
    <td><code>@RdfProperty</code></td><td>Fields, Methods</td><td>Value to define another property URI</td>
  </tr>
  <tr>
    <td><code>@RdfType</code></td><td>Classes</td><td>Value to define a rdfs:type property on the resource</td>
  </tr>
  <tr>
    <td><code>@Transient</code></td><td>Fields</td><td>Indicate that this field must not be converted</td>
  </tr>
  <tr>
    <td><code>@SameAs</code></td><td>Enum Constants</td><td>Value to define a owl:sameAs property on the resource</td>
  </tr>
  <tr>
    <td><code>@Label</code></td><td>Fields, Methods</td><td>Value to define a rdfs:label property on the resource</td>
  </tr>
</table>

## Future Work

Some ideas for the future development:

* Implement OWL binding
* Increase the configuration flexibility

If something is amiss, feel free to open an issue or make a pull request. The implementation is lightweight and allows to change the functionality very quickly.

[JenaBean]: http://code.google.com/p/jenabean/
[Jena API]: http://jena.apache.org/
[JAR]: https://github.com/platzhirsch/jtriple/raw/master/repository/releases/berlin/reiche/jtriple/jtriple/0.1/jtriple-0.1.jar
