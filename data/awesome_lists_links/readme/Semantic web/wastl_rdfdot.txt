RDFDot
======

Tools for drawing graphs from RDF files with GraphViz implemented in Java 8 using Java Native Interface calls
to GraphViz.

# Demo #

You can access the demo at [our demo server](http://demo4.newmedialab.at/rdfdot/), it should be running most of
the time. Simply cut & paste a Turtle or RDF/XML document, optionally change the configuration and press "Render".

# Building #

RDFDot comes with several different renderers for accessing the Graphviz layouting library. There are two different
approaches:

* call the _dot_ process as external shell command from Java; requires graphviz to be installed and accessible on your server
* use the Java Native Interface library that is provided by RDFDot; faster but requires manual compilation

## External Shell Command ##

The first approach is simple to build using standard Maven:

    mvn clean install -DskipTests

The -DskipTests is necessary at the moment, because some tests require the JNI library to work properly.

## Java Native Interface ##

If you want to use the JNI library, please follow the following sequence:

    mvn clean
    cd rdfdot-core/src/main/native
    make
    cd ../../../..
    mvn install

Calling make will download all the necessary C libraries, compile them and link them statically into a JNI library
for RDFdot. Note that this has only been tested to work on Linux.


# Installing #

RDFDot currently consists of the libraries rdfdot-api and rdfdot-core, which you can use in your own projects, and the
web application rdfdot-web, which can be deployed in any Java web container. When RDFDot has been properly installed,
simply add the approprate Maven dependencies to your project:

    <dependency>
        <groupId>net.wastl.rdfdot</groupId>
        <artifactId>rdfdot-core</artifactId>
        <version>1.0.0-SNAPSHOT</version>
    </dependency>

If you want to use the (faster) JNI rendering library, it is necessary that you copy the libgraphviz.so library to
an appropriate location and call Java with -Djava.library.path=/path/to/location.


# Library Usage #

## Configuration ##

The rdfdot-api library contains a class GraphConfiguration, which can be used for setting layouting configuration
options for the graph. It currently supports changing the node style, shape, color, and fillcolor for URI, BNode and
Literal nodes, as well as the arrow shape, style and color for edges. Furthermore, it is possible to change the layout
direction of the graph (default: left-right). All available options are defined using the enums Arrows, Layouts, Shapes
and Styles.

## RDFHandler ##

The visualization library is implemented as a Sesame RDFHandler, so it can be used anywhere Sesame accepts an
RDFHandler, e.g.

* in a RIO parser (using RDFParser.setRDFHandler(...))
* in a repository query (using RepositoryConnection.exportStatements(...))
* in a SPARQL graph query (using GraphQuery.evaluate(...))

To initialize the RDFHandler, use e.g. the following sequence of statements:

        GraphConfiguration configuration = new GraphConfiguration();
        GraphvizSerializer serializer = new GraphvizSerializerNative(configuration);

        RDFParser parser = Rio.createParser(RDFFormat.TURTLE);
        parser.setRDFHandler(new GraphvizHandler(serializer));
        parser.parse(in, "http://localhost/");

        byte[] image = serializer.getResult();

Different GraphSerializers are available, including GraphSerializerNative (using JNI calls) and GraphSerializerCommand
(executing a shell command).

