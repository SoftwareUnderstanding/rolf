GLEEN - A regular path library for ARQ SparQL

The GLEEN library is a property function library for the Jena ARQ SparQL query engine.

GLEEN was developed by:

> Todd Detwiler  
> Structural Informatics Group  
> University of Washington

GLEEN is currently licensed under the Apache License, version 2.0

------

This version of GLEEN is forked from the last released version: 0.6.1

The fork was created by Victor J. Orlikowski (Duke University) in support of
the ORCA project (https://github.com/RENCI-NRIG/orca5), which makes use of GLEEN.

This new revision ports GLEEN forward to currently supported versions of JENA
(http://jena.apache.org/). There are two branches in the code - master branch tied to Jena 2.11.0 and producing Gleen artifact with version 0.6.3-jena-2.11.0-SNAPSHOT. The other branch is called JENA_3_3_0 and produces an artifact with version 0.6.3-jena-3.3.0-SNAPSHOT. 

------

### Building

For this revision of GLEEN, simply check out the source, make sure you have a
recent version of maven, and type:

mvn clean compile
