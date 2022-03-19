jvmrdftools
===========

Simple collection of RDF tools in scala

Important Notes
---------------
This software is the product of research carried out at the [University of Zurich](http://www.ifi.uzh.ch/ddis.html) and comes with no warranty whatsoever. Have fun!

TODO's
------
* unit tests
* documentation
* examples

Gotcha's
--------
Every time the project version information is changed, BuildInfo needs to be regenerated. To do that simply run:

```sh
$ sbt compile
```

to generate the assembly (used by the [rdftools](https://github.com/cosminbasca/rdftools) python module) simply run

```sh
$ sbt assembly
```

Thanks a lot to
---------------
* [University of Zurich](http://www.ifi.uzh.ch/ddis.html) and the [Swiss National Science Foundation](http://www.snf.ch/en/Pages/default.aspx) for generously funding the research that led to this software.
