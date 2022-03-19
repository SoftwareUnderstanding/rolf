morph-starter
=============

Getting started with morph. this project is a simple Java (and Scala) demo of how to use morph.

Currently it shows how to generate RDF data from relational databases, using an [R2RML](http://www.w3.org/TR/r2rml/) mapping.

**Requirements**
* Java7
* Sbt 0.13 (or maven)

**Running**

To run the example, download the code and run Sbt:

```
>sbt run
```

It will run the main method that has a configured small HSQLDB memory database, and uses predefined mappings.
The RDF is output to the console. 
You can check the `DemoQueryJava` file to tweak and change whaterver you want.

The script to create the test DB (.sql) and the mappings (.r2rml) are is in `src/main/resources/data`. 
The database jdbc config is in `src/main/resources/application.conf`

**Eclipse**

If you want to use Eclipse, you can type the following to generate the .project files:
```
sbt eclipse
```

This will generate the necessary Eclipse project files, classpath dependencies, etc. Then you can import the project in your Eclipse installation.
If you plan to use Scala we recommend installing the Scala IDE plugin.

**Maven**

If you prefer to use maven instead of sbt, there is a pom.xml file available. Otherwise you can just ignore its existence.
You can compile the code as usual: `mvn compile`, import it to Eclipse using the m2e plugin, etc.
