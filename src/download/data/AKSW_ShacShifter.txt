# The ShacShifter

[![Travis CI Build Status](https://travis-ci.org/AKSW/ShacShifter.svg)](https://travis-ci.org/AKSW/ShacShifter/)
[![Coverage Status](https://coveralls.io/repos/github/AKSW/ShacShifter/badge.svg?branch=master)](https://coveralls.io/github/AKSW/ShacShifter?branch=master)

The *ShacShifter* is a shape shifter for the [*Shapes Constraint Language (SHACL)*](https://www.w3.org/TR/shacl/) to various other format.
Currently our focus is on convertig a SHACL NodeShape to an [RDForms template](http://rdforms.org/#!templateReference.md).

## Installation and Usage

You have to install the python dependencies with `pip install -r requirements.txt`.

To run start with:

    $ bin/ShacShifter --help
    usage: ShacShifter [-h] [-s SHACL] [-o OUTPUT] [-f {rdforms,wisski,html}]

    optional arguments:
      -h, --help            show this help message and exit
      -s SHACL, --shacl SHACL
                            The input SHACL file
      -o OUTPUT, --output OUTPUT
                            The output file
      -f {rdforms,wisski,html}, --format {rdforms,wisski,html}
                            The output format
