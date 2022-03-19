PRISSMA
===========
### Context-Aware Adaptation for Linked Data

[PRISSMA](http://wimmics.inria.fr/projects/prissma) is a presentation-level framework for [Linked Data](http://linkeddata.org) adaptation.

It is a Java rendering engine for [RDF](http://www.w3.org/TR/rdf11-primer/) that selects the most appropriate presentation of RDF triples according to [mobile context](http://en.wikipedia.org/wiki/Context_awareness).

PRISSMA is compatible with the [Fresnel vocabulary](http://www.w3.org/2005/04/fresnel-info/manual/) and is based on a graph edit distance algorithm that finds optimal error-tolerant subgraph isomorphisms between RDF context graphs.

PRISSMA is optimized for Android platforms, but can be used in regular Java Projects as well.


# Installation

PRISSMA is a Java library, optimized for Android applications.

## Minimum Requirements

+ Java 1.6
+ Android 4.2.2

## Download
The latest PRISSMA release is [v1.0.0](https://github.com/lukostaz/prissma/releases/tag/v1.0.0). Download it and add it to your build path.

Make sure `config.properties` is included in your build path and that it contains the  parameter values needed by the search algorithm (e.g. [similarity threshold, cost of edit distance operations, etc](http://2014.eswc-conferences.org/sites/default/files/papers/paper_81.pdf)).


## Build from Source
Check out sources:

	$ git clone https://github.com/lukostaz/prissma.git

PRISSMA depends on the [Simmetrics](https://github.com/Simmetrics/simmetrics) and [WS4J](https://code.google.com/p/ws4j/) libraries, that are not available in Maven central. Simmetrics 1.6.2 and WS4J 1.0.1 are provided under `libs/`. You can install it in your local Maven repository:
	
    $ cd prissma
    
    $ mvn install:install-file \
          -Dfile=libs/simmetrics-1.6.2.jar \
          -DgroupId=simmetrics \
          -DartifactId=simmetrics \
          -Dversion=1.6.2 \
          -Dpackaging=jar

    $ mvn install:install-file \
          -Dfile=libs/ws4j-1.0.1.jar \
          -DgroupId=ws4j \
          -DartifactId=ws4j \
          -Dversion=1.0.1 \
          -Dpackaging=jar

PRISSMA depends on the `xerces-impl` Android-optimized version available [available here](http://elite.polito.it/index.php/research/downloads/182-jena-on-android-download). A copy of the library is available under `libs/`.
Add it to local Maven repository:

	$ mvn install:install-file \
          -Dfile=libs/xercesImpl-repack.jar \
          -DgroupId=xerces \
          -DartifactId=xercesImpl-repack \
          -Dversion=1.0.0 \
          -Dpackaging=jar


Build and install the library in local Maven repository:
	
    $ mvn install

Add the following dependency in your `pom.xml`:

	<dependency>
        <groupId>fr.inria.wimmics</groupId>
        <artifactId>PRISSMA</artifactId>
        <version>1.0.0</version>
    </dependency>


# Designing Prisms

Prisms are context-aware presentation metadata for Linked Data visualization based on Fresnel and PRISSMA.

Prisms can be created manually, or with [PRISSMA Studio](http://luca.costabello.info/prissma-studio/).

## Example

Prism to style a `dbpedia:Museum` when a user is walking in Paris.

First, define the Prism general information:	
```turtle
:museumPrism a prissma:Prism ;
   fresnel:purpose :walkingInParis ;
   fresnel:stylesheetLink  <style.css>.
```

Add some Fresnel Lenses:
```turtle
:museumlens a fresnel:Lens;
   fresnel:group :museumPrism;
   fresnel:classLensDomain dbpedia:Museum;
   fresnel:showProperties (  
                     dbpprop:location 
                     dbpprop:publictransit 
                     ex:telephone
                     ex:openingHours
                     ex:ticketPrice ) .
```

Add Fresnel styling metadata:

```turtle
:addressFormat a fresnel:Format ;
   fresnel:group :museumPrism ;
   fresnel:propertyFormatDomain 
                     dbpprop:location ;
   fresnel:label "Address" ;
   fresnel:labelStyle 
       "css-class1"^^fresnel:styleClass ;
   fresnel:valueStyle 
       "css-class2"^^fresnel:styleClass .
```

Finally, define a `prissma:Context` entity with the [PRISSMA vocabulary](http://ns.inria.fr/prissma/v2/prissma_v2.html):
```turtle
# PRISSMA context description
:walkingInParisArtLover a prissma:Context ;
   prissma:user :artLover ; 
   prissma:environment :parisWalking .
    
:artLover a prissma:User ;
   foaf:interest "art".

:parisWalking a prissma:Environment ;
   prissma:poi :paris ;
   prissma:motion "walking" .
	
:paris geo:lat "48.8567" ;
   geo:long "2.3508" ;
   prissma:radius "5000" .
```
Save the Prism locally, and store it in the Decomposition structure as explained below.



# API Overview

Make sure `config.properties` is included in your build path and that it contains the  parameters values needed by the search algorithm (e.g. [similarity threshold, cost of edit distance operations, etc](http://2014.eswc-conferences.org/sites/default/files/papers/paper_81.pdf)).

## Step 1: Decomposing Prisms

```java
// Instantiate a decomposer
Decomposer decomposer = new Decomposer();
// The Decomposition is the shared data structure for Prisms
Decomposition decomp = new Decomposition();

// The inputPrism Jena Model is the Prism read from local repository
Model inputPrism = ModelFactory.createDefaultModel();

// Get the path of the Prism local repository on Android devices.
// If executed on desktop environment p is a regular file path string.
String p = Environment.getExternalStorageDirectory().getAbsolutePath();
InputStream in = FileManager.get().open( p + "/PRISSMA/prisms/prism.ttl" );

// Decompose the Prism
if (in != null) {
    inputPrism.read(in, null,  "TURTLE");
    decomp = decomposer.decompose(inputPrism, decomp);
}

```

## Step 2: Running the search algorithm

```java
// Read input context
Model actualCtx = ModelFactory.createDefaultModel();
InputStream inCtx = FileManager.get().open( p + "/PRISSMA/ctx/ctx1.ttl" );
if (inCtx != null) {
    actualCtx.read(inCtx, null,  "TURTLE");
}

// Instantiate an error-tolerant matcher with a decomposition
Matcher matcher = new Matcher(decomp);
// get the prissma:Context element, i.e. the root element of input context
RDFNode ctxRoot = ContextUnitConverter.getRootCtxNode(actualCtx);
// Covnert core PRISSMA entities to their PRISSMA classes
ctxRoot = ContextUnitConverter.switchToClasses(ctxRoot, decomp);
// Execute error-tolerant match against Prisms in the decomposition
matcher.search(ctxRoot);

```

## Step 3: Rendering resources

```java
// inputResource is the desired RDF resource to display.
Model prism = readPrismFromFS(matcher.results);
Renderer r = new Renderer();
String html = r.renderHTML(prism, inputResource, true);

```


# Publications


+ L. Costabello. [Error-Tolerant RDF Subgraph Matching for Adaptive Presentation of Linked Data on Mobile](http://2014.eswc-conferences.org/sites/default/files/papers/paper_81.pdf). 11th Extended Semantic Web Conference (ESWC), 2014.
+ L. Costabello. [PRISSMA, Towards Mobile Adaptive Presentation of the Web of Data](http://iswc2011.semanticweb.org/fileadmin/iswc/Papers/DC_Proposals/70320273.pdf). ISWC 2011 Doctoral Consortium, Bonn, Germany

# Licence
	
    Copyright (C) 2013-2015 Luca Costabello, v1.0.0

    This program is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation; either version 2 of the License, or (at your
    option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, see <http://www.gnu.org/licenses/>.

# Contacts
Further details on the [PRISSMA Project Page](http://wimmics.inria.fr/projects/prissma/), or contact [Luca Costabello](http://luca.costabello.info).

