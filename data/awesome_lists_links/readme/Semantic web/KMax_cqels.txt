# CQELS (Continuous Query Evaluation over Linked Data)

This repository is a fork of https://code.google.com/p/cqels/ repository on Google Code.

_DISCLAMER: I'm not a developer of the original CQELS._

## Install

Add the following repository to your pom.xml:
```
<repository>
    <id>cqels.mvn-repo</id>
    <url>https://raw.github.com/KMax/cqels/mvn-repo/</url>
    <snapshots>
        <enabled>true</enabled>
        <updatePolicy>always</updatePolicy>
    </snapshots>
</repository>
```

and declare the following dependency:
```
<dependency>
    <groupId>org.deri.cqels</groupId>
    <artifactId>cqels</artifactId>
    <version>...</version>
</dependency>
```

## Releases
### 1.0.0
* Mavenized build,
* Support of [BIND](http://www.w3.org/TR/sparql11-query/#bind) operator,
* Initial support of remote SPARQL endpoints via [SPARQL Graph Protocol](http://www.w3.org/TR/sparql11-http-rdf-update/),
* Fixed an FileException (at ObjectFileStorage) exception _([commit](https://github.com/KMax/cqels/commit/4382fe7e2f15a8c205a47ab3cd0e25842e558c30))_.

### 1.1.0
* Updated [Jena TDB](https://jena.apache.org/documentation/tdb/) up to 1.1.2 version

Code license: [LGPLv3.0](https://github.com/KMax/cqels/blob/master/LICENSE)
