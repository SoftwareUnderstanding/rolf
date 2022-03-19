# Rdf.Vocabularies [![Build status](https://ci.appveyor.com/api/projects/status/8uepsle8g54v101m/branch/master?svg=true)](https://ci.appveyor.com/project/tpluscode78631/rdf-vocabularies/branch/master) [![NuGet version](https://badge.fury.io/nu/rdf.vocabularies.svg)](https://badge.fury.io/nu/rdf.vocabularies) [![codefactor][codefactor-badge]][codefactor-link]
 
## What 

Statically typed RDF Vocabularies for .NET. Most generated automatically from respective RDF/OWL files:

* `bibo` - [Bibliographic Ontology](http://lov.okfn.org/dataset/lov/vocabs/bibo)
* `dc` - [Dublin Core Metadata Element Set](http://lov.okfn.org/dataset/lov/vocabs/dce)
* `dcterms` - [DCMI Metadata Terms](http://lov.okfn.org/dataset/lov/vocabs/dcterms)
* `foaf` - [Friend of a Friend Vocabulary](http://lov.okfn.org/dataset/lov/vocabs/foaf)
* `Hydra` - [A lightweight vocabulary for hypermedia-driven Web APIs](http://www.hydra-cg.com/)
* `opus` - [Ontology of Computer Science Publications](http://lov.okfn.org/dataset/lov/vocabs/opus)
* `owl` - [Ontology Web Language](http://lov.okfn.org/dataset/lov/vocabs/owl)
* `rdf` - [The RDF Concepts Vocabulary](http://lov.okfn.org/dataset/lov/vocabs/rdf)
* `rdfs` - [The RDF Schema vocabulary](http://lov.okfn.org/dataset/lov/vocabs/rdfs)
* `schema` - [Schema.org vocabulary](http://lov.okfn.org/dataset/lov/vocabs/schema)
* `sioc` - [Semantically-Interlinked Online Communities](http://lov.okfn.org/dataset/lov/vocabs/sioc)
* `skos` - [Simple Knowledge Organization System](http://lov.okfn.org/dataset/lov/vocabs/skos)

## How

Install NuGet

```
install-package Rdf.Vocabularies
```

Use

``` csharp
using Vocab;

var rdfsLabel = Rdfs.label;
```

[codefactor-badge]: https://www.codefactor.io/repository/github/wikibus/Rdf.Vocabularies/badge/master
[codefactor-link]: https://www.codefactor.io/repository/github/wikibus/Rdf.Vocabularies/overview/master
