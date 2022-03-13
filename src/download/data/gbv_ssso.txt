
This repository contains the specification of **Simple Service Status Ontology
(SSSO)**.

The URI of this ontology is <http://purl.org/ontology/ssso> and it's URI
namespace is <http://purl.org/ontology/ssso#>.

See <http://gbv.github.io/ssso> for a full documentation. RDF serializations
are available at <http://gbv.github.io/ssso/ssso.ttl> and
<http://gbv.github.io/ssso/ssso.owl>.

[Feedback](https://github.com/gbv/ssso/issues) is welcome!

# Overview

The following diagram illustrates the classes and properties definied in this ontology:

``` {.ditaa}
    nextService / previousService
               ------
              |      |
              v      v
       +--------------------+
       |    ServiceEvent    |
       |                    |
       |   ReservedService  |
       |   PreparedService  |
       |   ProvidedService  |
       |   ExecutedService  |
       |   RejectedService  |
       |                    |
       | ServiceFulfillment |
       +-----^--------------+
             |      ^
             |      |
              ------
dcterms:hasPart / dcterms:partOf
```

