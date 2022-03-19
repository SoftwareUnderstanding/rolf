# carml-cli
Interface for CARML library. At this moment works only with xml files.

#How to build
Build with maven
```
mvn clean package
```

#How to use
- Possible to convert a single file:
```
java -jar cli-0.0.1-SNAPSHOT-jar-with-dependencies.jar -i inputfile.xml -m rml.mapping.ttl -o output
```

- Possible to convert a folder (After the conversion of each file the output is streamed to the rdf output file):
```
java -jar cli-0.0.1-SNAPSHOT-jar-with-dependencies.jar -f /folder -m /rml.mapping.ttl -o /output.ttl
```

- Adding output format:
```
java -jar cli-0.0.1-SNAPSHOT-jar-with-dependencies.jar -f /folder -m /rml.mapping.ttl -o /output.nt -of nt
```

- Adding mapping format:
```
java -jar cli-0.0.1-SNAPSHOT-jar-with-dependencies.jar -f /folder -m /rml.mapping.ttl -o /output.nt -mf nt
```

