[![Build Status](https://travis-ci.org/stoewer/fluent-sparql.png?branch=master)](https://travis-ci.org/stoewer/fluent-sparql)
[![Coverage Status](https://coveralls.io/repos/stoewer/fluent-sparql/badge.png?branch=master)](https://coveralls.io/r/stoewer/fluent-sparql?branch=master)

## About Fluent SPARQL

This project aims to provide an interface for creating and executing SPARQL queries using a DSL-like fluent API that
allows to write code that resembles the SPARQL syntax, such as:

```java
Query q = sparql.select("name")
                    .add("?a", RDF.type, FOAF.Person)
                    .add("?a", FOAF.givenname, "Frodo")
                    .add("?a", FOAF.family_name, "?name")
                    .filter("?name").regexp("^Baggins$")
                .orderByAsc("name")
                .limit(1)
                .query();
```

## Status

This project is still work in progress and at this time there is no stable release!
