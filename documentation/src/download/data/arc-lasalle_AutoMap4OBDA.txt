# AutoMap4OBDA
AutoMap4OBDA is a system which automatically generates R2RML mappings based on the intensive use of relational source contents and features of the target ontology. AutoMap4OBDA takes as inputs a relational database (i.e., PostgreSQL) and an ontology in OWL to produce a putative ontology from a relational database which is used as an intermediate element in the relational-to-ontology process. Moreover, AutoMap4OBDA has been designed to be used in OBDA scenarios and is able to generate fully compliant R2RML mappings without user intervention.

In AutoMap4OBDA, the database content and features of the target ontology are taken into account during the mapping generation process. We have developed three techniques that make the mapping process strongly dependent on the input database and the features of the target ontology to increase the performance of the relational-to-ontology mappings.

- Ontology learning technique to infer class hierarchies for development of a putative ontology
- String similarity metric selection based on target ontology labels for ontology alignment
- Short path strategy for R2RML mapping generation based on alignments

This is the first version which will require some code cleansing and refactoring. Currently it is only supporting PostgreSQL.

Use:
```
java -jar automap4obda.jar -db <databaseURL> -schema <schemaname> -driver <databaseDriver> -u <username> - p "<password>" 
-n <ontologyname> -d <path-to-domainontology> -o <outputfiles> 
[-attrasclass <0/1>] [-ol <0/1>] [-olclasstable <0/1>] [-olclassnamealone <0/1>] [-extendedmappings <0/1>]
```

Example:
```
java -jar automap4obda.jar -db jdbc:postgresql:postgres -schema sigkdd_structured -driver org.postgresql.Driver -u postgres -p "postgres" 
-n sigkdd_structured_putative -d "c:\...\sigkdd_structured.ttl" -o sigkdd_structured_putative 
-attrasclass 1 -ol 1 -olclasstable 1 -olclassnamealone 1  -extendedmappings 1
```


- Password can be empty
- attrasclass: Attributes as classes option (default 1)
- ol: Ontology learning technique option (default 1)
- olclasstable: New classes in ontology learning technique belongs to the table class (1) or to the column class (0) (default 1)
- olclassnamealone: New classes in ontology learning technique is taken as it is in the DB (1) or the name of the table/column is attached (0) (default 1)
- extendedmappings: Short path technique option (default 1)

http://arc.salleurl.edu/automap4obda

Copyright (C) 2016 ARC Engineering and Architecture La Salle, Ramon Llull University.
 
for comments please contact Alvaro Sicilia (ascilia@salleurl.edu)

