rdftools
========

rdftools is a python wrapper over a number of RDF related tools
* rdf parsers / serializers
* void utilities
* lubm generator
* etc

Important Notes
---------------
This software is the product of research carried out at the [University of Zurich](http://www.ifi.uzh.ch/ddis.html) and comes with no warranty whatsoever. Have fun!

TODO's
------
* The project is not documented (yet)

How to Compile/Install the Project
----------------------------------
Ensure that *libraptor2* v2.0.13+ and *cityhash* are installed on your system (either using the package manager of the OS or compiled from source).

To install **rdftools** you have two options: 1) manual installation (install requirements first) or 2) automatic with **pip**

**Manual** installation:
```sh
$ git clone https://github.com/cosminbasca/rdftools
$ cd rdftools
$ python setup.py install
```

Install the project with **pip**:
```sh
$ pip install https://github.com/cosminbasca/rdftools
```

Also have a look at the build.sh, clean.sh, test.sh scripts included in the codebase 

To include the latest JVM RDF tools update to the latest of [jvmrdftools](https://github.com/cosminbasca/jvmrdftools) and create an assembly:

```sh
$ sbt compile assembly
```

copy the resulting jar from the target folder to the *lib* folder inside the *rdftools.tools.jvmrdftools* module and reinstall the python package.


The tools
---------

To find out what a tool does, simply supply the *--help* comand line argument to any of the tools
Available tools:

* rdfconvert, convert RDF files from source format to a destination format using the *libraptor2* C RDF parser

```sh
usage: rdfconvert [-h] [--clear] [--dst_format DST_FORMAT]
                  [--buffer_size BUFFER_SIZE] [--version]
                  SOURCE

rdftools v0.9.2, rdf converter, based on libraptor2

positional arguments:
  SOURCE                the source file or location (of files) to be converted

optional arguments:
  -h, --help            show this help message and exit
  --clear               clear the original files (delete) - this action is
                        permanent, use with caution!
  --dst_format DST_FORMAT
                        the destination format to convert to. Supported
                        parsers: ['rdfxml', 'ntriples', 'turtle', 'trig',
                        'guess', 'rss-tag-soup', 'rdfa', 'nquads', 'grddl'].
                        Supported serializers ['rdfxml', 'rdfxml-abbrev',
                        'turtle', 'ntriples', 'rss-1.0', 'dot', 'html',
                        'json', 'atom', 'nquads'].
  --buffer_size BUFFER_SIZE
                        the buffer size in Mb of the input buffer (the parser
                        will only parse XX Mb at a time)
  --version             the current version
```

* rdfconvert2 convert RDF files from source format to a destination format using the *rdf2rdf* java RDF parser

```sh
usage: rdfconvert2 [-h] [--clear] [--dst_format DST_FORMAT]
                   [--workers WORKERS] [--version]
                   SOURCE

rdftools v0.9.2, rdf converter (2), makes use of rdf2rdf bundled - requires
java

positional arguments:
  SOURCE                the source file or location (of files) to be converted

optional arguments:
  -h, --help            show this help message and exit
  --clear               clear the original files (delete) - this action is
                        permanent, use with caution!
  --dst_format DST_FORMAT
                        the destination format to convert to
  --workers WORKERS     the number of workers (default -1 : all cpus)
  --version             the current version
```

* rdfencode, endode an ntriples file to a binary format (each S, P, O string is hashed with *cityhash* 64 bit)

```sh
usage: rdfencode [-h] [--version] SOURCE

rdftools v0.9.2, encode the RDF file(s)

positional arguments:
  SOURCE      the source file or location (of files) to be encoded

optional arguments:
  -h, --help  show this help message and exit
  --version   the current version
```

* genlubm, generate a **LUBM** dataset (in parallel)

```sh
usage: genlubm [-h] [--univ UNIV] [--index INDEX] [--seed SEED]
               [--ontology ONTOLOGY] [--workers WORKERS] [--version]
               OUTPUT

rdftools v0.9.2, lubm dataset generator wrapper (bundled) - requires java

positional arguments:
  OUTPUT               the location in which to save the generated
                       distributions

optional arguments:
  -h, --help           show this help message and exit
  --univ UNIV          number of universities to generate
  --index INDEX        start university
  --seed SEED          the seed
  --ontology ONTOLOGY  the lubm ontology
  --workers WORKERS    the number of workers (default -1 : all cpus)
  --version            the current version
```

* genlubmdistro generate a **LUBM** dataset (in parallel) and mix the universities to *N* sites with the specified distribution

```sh
usage: genlubmdistro [-h] [--distro DISTRO] [--univ UNIV] [--index INDEX]
                     [--seed SEED] [--ontology ONTOLOGY] [--pdist PDIST]
                     [--sites SITES] [--clean] [--workers WORKERS] [--version]
                     OUTPUT

rdftools v0.9.4, lubm dataset generator wrapper (bundled) - requires java

positional arguments:
  OUTPUT               the location in which to save the generated
                       distributions

optional arguments:
  -h, --help           show this help message and exit
  --distro DISTRO      the distibution to use, valid values are ['seedprop',
                       'uni2many', 'horizontal', 'uni2one']
  --univ UNIV          number of universities to generate
  --index INDEX        start university
  --seed SEED          the seed
  --ontology ONTOLOGY  the lubm ontology
  --pdist PDIST        the probabilities used for the uni2many distribution,
                       valid choices are ['3S', '7S', '5S'] or file with
                       probabilities split by line
  --sites SITES        the number of sites
  --clean              delete the generated universities
  --workers WORKERS    the number of workers (default -1 : all cpus)
  --version            the current version
```
* genvoid, generate [VoID](http://www.w3.org/TR/void/) statistics from the source file

```sh
usage: genvoid [-h] [--version] SOURCE

rdftools v0.9.2, generate void statistics for RDF source file

positional arguments:
  SOURCE      the source file to be analized

optional arguments:
  -h, --help  show this help message and exit
  --version   the current version
```

* genvoid2, generate [VoID](http://www.w3.org/TR/void/) statistics from the RDF source file, using the *nxparser VoID* exporter

```sh
usage: genvoid2 [-h] [--dataset_id DATASET_ID] [--use_nx] [--version] SOURCE

rdftools v0.9.2, generate a VoiD descriptor using the nxparser java package

positional arguments:
  SOURCE                the source file to be analized

optional arguments:
  -h, --help            show this help message and exit
  --dataset_id DATASET_ID
                        dataset id
  --use_nx              if true (default false) use the nx parser builtin void
                        generator
  --version             the current version
```

* ntround, round all numeric literals (typed or untyped) in an ntriples files with the given precision

```sh
usage: ntround [-h] [--prefix PREFIX] [--precision PRECISION] [--version] PATH

rdftools v0.9.2, rounds ntriple files in a folder, (rounds the floating point literals)

positional arguments:
  PATH                  location of the indexes

optional arguments:
  -h, --help            show this help message and exit
  --prefix PREFIX       the prefix used for files that are transformed, cannot
                        be the enpty string!
  --precision PRECISION
                        the precision to round to, if 0, floating point
                        numbers are rounded to long
  --version             the current version
```

Thanks a lot to
---------------
* [University of Zurich](http://www.ifi.uzh.ch/ddis.html) and the [Swiss National Science Foundation](http://www.snf.ch/en/Pages/default.aspx) for generously funding the research that led to this software.
