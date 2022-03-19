[1]: ./src/main/java/org/aksw/fox/binding/java/Examples.java

fox-java
========

Java bindings for FOX - Federated Knowledge Extraction Framework


In [Examples.java][1] you can find an example.

```Java
final IFoxApi fox = new FoxApi()
     .setApiURL(new URL("http://0.0.0.0:4444/fox"))
     .setTask(FoxParameter.TASK.RE)
     .setOutputFormat(FoxParameter.OUTPUT.JSONLD)
     .setLang(FoxParameter.LANG.EN)
     .setInput("A. Einstein was born in Ulm.")
     // .setLightVersion(FoxParameter.FOXLIGHT.ENBalie)
     .send();

 final String jsonld = fox.responseAsFile();
 final FoxResponse response = fox.responseAsClasses();

 List<Entity> entities = response.getEntities();
 List<Relation> relations = response.getRelations();

```

### Maven
    <dependencies>
      <dependency>
        <groupId>com.github.renespeck</groupId>
        <artifactId>fox-java</artifactId>
        <version>e67a2bd475</version>
      </dependency>
    </dependencies>

    <repositories>
        <repository>
            <id>jitpack.io</id>
            <url>https://jitpack.io</url>
        </repository>
    </repositories>
