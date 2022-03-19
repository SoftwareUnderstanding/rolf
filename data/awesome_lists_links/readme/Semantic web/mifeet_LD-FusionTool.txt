LD-FusionTool
==========
###Data Fusion and Conflict Resolution tool for Linked Data



LD-FusionTool is a standalone tool (and [a module](https://github.com/mifeet/FusionTool-DPU) for [UnifiedViews](http://unifiedviews.eu) ETL framework) executing the Data Fusion and Conflict Resolution steps in the integration process for RDF, where data are merged to produce consistent and clean representations of objects, and conflicts which emerged during data integration need to be resolved.

**Please visit [the official page of LD-FusionTool](http://mifeet.github.io/LD-FusionTool/) for more information about what LD-FusionTool is, how it works and how you can download it and run.**



Building from sources
========

In order to use LD-FusionTool, download the sources, build them with Maven (run <code>mvn clean install</code> in the <code>sources</code> directory of the project). Locate the built binaries in <code>sources/odcsft-application/target</code> and execute<br/> <code>java -jar odcsft-application-&lt;version&gt;-executable.jar &lt;configuration-file&gt;.xml</code>. Running the executable without parameters shows more usage options and sample configuration files can be found at <a href="https://github.com/mifeet/LD-FusionTool/tree/master/examples">examples</a> (file <code>sample-config-full.xml</code> serves as the working documentation of the configuration file).


