Stardog Groovy
==========

Licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)  
_Current Version **5.3.5**_ 

Stardog Groovy - Groovy language bindings to use to develop apps with the [Stardog Graph / RDF Database](http://stardog.com).  

![Stardog](http://stardog.com/img/stardog.png)   

## What is it? ##

This bindings provides a set of idiomatic Groovy APIs for interacting with the Stardog database, similar to the Stardog Spring project - an easy to use method for creating connection pools, and the ability run queries over them. To run the queries, Stardog Groovy uses standard Groovy patterns, such as passing in a closure to iterate over result sets.  Common use cases for Stardog-groovy are ETL scripts, command line applications, usage with Grails, or other Groovy frameworks.   

## How to use it

1. Download Stardog from stardog.com, and follow the installation instructions
2. Add the `com.complexible.stardog:stardog-groovy:<version>` dependency declaration to your build tool, such as Maven or Gradle
3. Make sure your build prioritizes your local Maven repository (i.e. `~/.m2/repository`), where the core Stardog binaries were installed by step 2
4. Enjoy!

There is also a `shadowJar` task available via the Shadow plugin to produce a fatjar with all of the Stardog dependencies.

## Quickstart

```
@Grab('com.complexible.stardog:stardog-groovy:5.3.5')
import com.complexible.stardog.ext.groovy.Stardog

def stardog = new Stardog(url: "http://localhost:5820", to:"testdb", username: "admin", password:"admin", reasoning: true)

stardog.query("select ?s ?p ?o where { ?s ?p ?o } limit 2", { println it })
```

## Examples ##

Create a new embedded database in one line
```groovy
	def stardog = new Stardog(home:"/opt/stardog", to:"testgroovy", username:"admin", password:"admin")
```

Collect query results via a closure
```groovy
	def list = []
	stardog.query("select ?x ?y ?z WHERE { ?x ?y ?z } LIMIT 2") { list << it } 
	// list has the two Sesame BindingSet's added to it, ie TupleQueryResult.next called per each run on the closure
```

Collect query results via projected result values
```groovy
    stardog.each("select ?x ?y ?z WHERE { ?x ?y ?z } LIMIT 2", {
       println x // whatever x is bound to in the result set
       println y // ..
       println z // 
    }
```

Like query, this is executed over each TupleQueryResult

Insert multidimensional arrays, single triples also works
```groovy
	stardog.insert([ ["urn:test3", "urn:test:predicate", "hello world"], ["urn:test4", "urn:test:predicate", "hello world2"] ])
```

Remove triples via a simple groovy list
```groovy
	stardog.remove(["urn:test3", "urn:test:predicate", "hello world"])
```

## Upgrading from Prior Releases

Significant changes in 2.1.3:

*    Installation now available via Maven Central and "com.complexible.stardog:stardog-groovy:2.1.3" dependency
*    No longer a dependency on Spring, i.e. the Stardog-Spring DataSource can no longer be passed as a constructor.  The Stardog Groovy class performs all the same operations.
*    Stardog-groovy 4.2.1 and later should be built with Gradle 2.3


## Development ##

To get started, just clone the project. You'll need a local copy of Stardog to be able to run the build. For more information on starting the Stardog DB service and how it works, go to [Stardog's documentation](http://stardog.com/docs/), where you'll find everything you need to get up and running with Stardog.

Once you have the local project, start up a local Stardog and create a testdb with `stardog-admin db create -n testdb $STARDOG/data/examples/lumbSchema.owl $STARDOG/data/examples/University0_0.owl`. 

You can then build the project

    gradle build    # validate all the test pass
    gradle install  # install jar into local m2

That will run all the JUnit tests and create the jar in build/libs.  The test does use a running Stardog, and if you receive error during the test it is likely you're Stardog server is not running or has an invalid license.  This usually manifests in an exit of a Gradle worker, which is the JVM running the JUnit tests. 


## Contributing ##

This framework is in continuous development, please check the [issues](https://github.com/clarkparsia/stardog-groovy/issues) page. You're welcome to contribute.

## License

Copyright 2015 - 2018 Stardog Union
Copyright 2012 - 2015 Clark & Parsia, LLC
Copyright 2012 Al Baker

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

* [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)  

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


