## SPARQL Template

[![Build status](https://travis-ci.org/gushakov/sparql-template.svg?branch=master)](https://travis-ci.org/gushakov/sparql-template)

Small library for traversing an RDF store using automatic mapping of triples to annotated POJOs.

## Highlights

 * Support of any store exposing HTTP SPARQL endpoint
 * Uses [Jena API](https://jena.apache.org/) to load and process RDF triples
 * Uses [MappingContext](https://github.com/spring-projects/spring-data-commons/blob/master/src/main/java/org/springframework/data/mapping/context/MappingContext.java) from Spring Data Commons to process class annotations
 * On-demand (lazy) loading of relations using automatic proxying with [ByteBuddy](http://bytebuddy.net/)
 * Easily extended for conversion from any `org.apache.jena.graph.Node` to a custom Java type
 * Some useful converters are registered by default, see `ch.unil.sparql.template.convert.ExtendedRdfJavaConverter`
  + `java.util.Date`
  + `java.time.ZonedDateTime`
  + `java.time.Duration`
  + `java.net.URL`
 
## Examples

Assume we want to retrieve some information about a person from the [DBPedia](http://dbpedia.org) using the [SPARQL endpoint](http://dbpedia.org/sparql).
We annotate our domain POJO as following.

```java
// marks this as an RDF entity
@Rdf
public class Person {

    // will be mapped from the value of http://dbpedia.org/ontology/birthName
    @Predicate(DBP_NS)
    private String birthName;

    // will be mapped from the value of http://www.w3.org/2000/01/rdf-schema#label for the Russian language
    @Predicate(value = RDFS_NS, language = "ru")
    private String label;

    // will be mapped from the value of http://dbpedia.org/property/birthDate, automatic conversion to java.time.ZonedDateTime
    @Predicate(DBP_NS)
    private ZonedDateTime birthDate;

    // will be mapped from the values of http://dbpedia.org/property/spouse, lazy load of relationships
    @Predicate(DBP_NS)
    @Relation
    private Collection<Person> spouse;
}
```

Then we can just use `ch.unil.sparql.template.SparqlTemplate` to load the triples from the DBPedia converting
them automatically to the required Java instance.

```java
    // get the default SPARQL template
    final SparqlTemplate sparqlTemplate = new SparqlTemplate("https://dbpedia.org/sparql");

    // load information about Angelina Jolie
    final Person person = sparqlTemplate.load(DBR_NS + "Angelina_Jolie", Person.class);

    System.out.println(person.getBirthName());
    // Angelina Jolie Voight

    System.out.println(person.getLabel());
    // Джоли, Анджелина

    System.out.println(person.getBirthDate().format(DateTimeFormatter.ofPattern("dd/MM/yyyy (EEE)", Locale.ENGLISH)));
    // 04/06/1975 (Wed)

    System.out.println(person.getSpouse().stream()
            .filter(p -> p.getBirthName() != null && p.getBirthName().contains("Pitt"))
            .findAny().get().getBirthName());
    // William Bradley Pitt

```
