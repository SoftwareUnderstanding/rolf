This repository contains the **Document Service Ontology (DSO)**

The URI of this ontology is going to be <http://purl.org/ontology/dso> and it's
URI namespace is going to be <http://purl.org/ontology/dso#> (not registered
yet).

The current version of this specification can be found at <http://gbv.github.io/dso/>
and a public git repository at <https://github.com/gbv/dso>.
[Feedback](https://github.com/gbv/dso/issues) is welcome!

The following diagram illustrates the classes and properties defined in this ontology.

~~~
    +---------------------+
    |  dso:ServiceEvent   |
    | +-----------------+ |  hasDocument    +-----------------------+
    | | DocumentService |------------------>| ...any document class |
    | |                 |<------------------|                       |
    | |  Loan           | |  hasService     +-----------------------+
    | |  Presentation   | |
    | |  Interloan      | |
    | |  OpenAccess     | |
    | |  Digitization   | |
    | |  Identification | |
    | |  ...            | |
    | +-----------------+ |
    +---------------------+
~~~


