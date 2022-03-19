*owl-me*
====

#### a Java-based module extractor for OWL ontologies ####

Built using the [OWL API](http://owlapi.sourceforge.net/). 


summary
--------------------
*owl-me* is a standalone tool designed to extract different types of [Locality-based modules](http://owl.cs.manchester.ac.uk/research/modularity/) from OWL ontologies.

The tool takes as inputs an ontology and a text file. The latter is the so-called *signature file*, which contains entity (class and object/data property) IRIs. The tool extracts a module for the specified set of IRIs (i.e. signature) onto a chosen location.


usage
--------------------
Build using the Ant script and run the **owl-me.jar** file. For large ontologies you may have to increase the heap space and entity expansion limit (esp. for ontologies in RDF/XML), e.g., for 4GB heap:<br><br>
`java -jar -Xmx4G -DentityExpansionLimit=100000000 owl-me.jar`


signatures for module extraction
--------------------
Signature files should contain entity IRIs as they appear in the original ontology. IRIs can be separated by any of the following delimiters:
  * Comma (e.g. CSV files)
  * White space
  * Vertical bar "|"
  * Tab
  * New line

The file may also contain headers or comments, so long as the line or part thereof is preceded with '%'. All text following '%' is ignored. Check the *example signature file contents* below:

% My header<br>
Class_IRI_1, Class_IRI_2 Class_IRI_3<br>
Property_IRI_2 | Property_IRI_3    % Main properties<br>
<br>
% Some comment<br>
Class_IRI_4<br>


SNOMED CT
--------------------
The module extractor accepts signature files for the SNOMED CT ontology in the *UMLS Core Subset format*. Any manually constructed signature files **should have the concept ID's delimited by vertical bars "|"**, in a similar way as the UMLS Core Subset files.


deployment
--------------------
The module extractor is compatible with **Java 1.6 and above**. It was tested with Java 1.7 and 1.8., and relies mainly on the following project:

 * [OWL API](http://owlapi.sourceforge.net/) (v4.0.1)


contact
--------------------
Consider checking the [OWL@Manchester](http://owl.cs.manchester.ac.uk) website (and linked publications) for more information regarding _Locality-based_ modules, before submitting queries.

If you come across any bugs please use the "Issues" tab to describe the problem, along with sufficient data to reproduce it (i.e. the ontology and signature used).