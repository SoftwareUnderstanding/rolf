# swift-hdt

## An HDT RDF Parser

### Build

On MacOS 10.14:

```
% swift build -Xswiftc "-target" -Xswiftc "x86_64-apple-macosx10.14"
```

On Linux:

```
% swift build
```

### Parse an HDT file

```
% ./.build/release/hdt-parse swdf-2012-11-28.hdt
_:b1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#_1> <http://data.semanticweb.org/person/barry-norton> .
_:b1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#_2> <http://data.semanticweb.org/person/reto-krummenacher> .
_:b1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Seq> .
_:b10 <http://www.w3.org/1999/02/22-rdf-syntax-ns#_1> <http://data.semanticweb.org/person/robert-isele> .
_:b10 <http://www.w3.org/1999/02/22-rdf-syntax-ns#_2> <http://data.semanticweb.org/person/anja-jentzsch> .
_:b10 <http://www.w3.org/1999/02/22-rdf-syntax-ns#_3> <http://data.semanticweb.org/person/christian-bizer> .
_:b10 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Seq> .
...
```

### Limitations

This project is early in development, and has many limitations:

* Only serializing the entire HDT file is possible (triple pattern matching is planned for the future)
* Only "Four Part" dictionary encoding is currently supported
* Only "Log64" encoding of bitmap triples values is currently supported

### References

* [HDT](http://www.rdfhdt.org)
* [HDT Binary Format](http://www.rdfhdt.org/hdt-binary-format/)
