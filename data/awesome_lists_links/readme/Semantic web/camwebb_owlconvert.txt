owlconvert
==========

> A very simple OWL format converter based on [OWLAPI](http://owlapi.sourceforge.net/2.x.x/index.html).

I wanted a local tool to handle format conversion similar to the
online converter at
[cs.Manchester](http://owl.cs.manchester.ac.uk/converter/), but
couldn’t find anything handy, ready made.  Before this hack, I knew
close to zero java, so please be lenient!

# Usage

      $ owlconvert  manchester|functional|turtle|rdfxml  <owl infile>

sends to standard out.

# Installation

To compile, place in the same dir as `owlapi-bin.jar`

      $ javac -classpath owlapi-bin.jar owlconvert.java

Run with (replacing with correct paths):

      $ java -classpath "owlapi-bin.jar:." owlconvert

Make a shell script or alias to run from anywhere:

      $ alias owlconvert 'java -classpath "$OWLAPI:$OWLCONVERTPATH" owlconvert'

