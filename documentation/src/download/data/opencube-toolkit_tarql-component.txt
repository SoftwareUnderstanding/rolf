OpenCube Toolkit - Tarql Extension
===============


[Tarql](https://github.com/cygri/tarql) is a command-line tool for converting CSV files to RDF using SPARQL 1.1 syntax. It's written in Java and based on Apache ARQ.

### How it works

TARQL Extension for data cubes is a tool for converting CSV files to RDF accordingly to SPARQL 1.1 syntax. The data cubes are generated based on the provided, easy to modify, query templates. The extension is integrated with the Information Workbench as a standard data provider. The interface delivered enables user to specify the basic information about the provider along with the location of the CSV file, polling intervals and to modify the cube mapping query via provided SPARQL editor. The interface includes information on status and the time that was needed to transform the data and enables to browse the output triples generated.

The TARQL extension can be stacked with other data components for instance the output RDF can be visualised with i.e. the OpenCube Map View.


###Functionality

The component is built on top of Apache ARQ2. The OpenCube TARQL component includes the new release of TARQL. It brings several improvements, such as: streaming capabilities, multiple query patterns in one mapping file, convenient functions for typical mapping activities, validation rules included in mapping file and increased flexibility (dealing with CSV variants like TSV).

The OpenCube Toolkit user is able to create Cube directly from the input files and store the output in the SPARQL endpoint. The OpenCube TARQL extension offers the following options:

+ Stream CSV processing
+ Use of column headers as variable names
+ Translate the CSV imported table into RDF by using a prepared mapping file (SPARQL construct schema)
+ Test the mapping file (shows only the CONSTRUCT template, variable names, and a few input rows)
