geonames
========

Simple utility scripts for downloading, unpacking and reformatting the Geonames 
RDF data dumps.

Author
------

Leigh Dodds (leigh.dodds@talis.com)

Download
--------

[http://github.com/ldodds/geonames]

Overview
--------

Geonames is a great resource for geographical information. Helpfully they 
publish data exports in a variety of formats, allowing others to process and 
manipulate the data locally.

Unfortunately the RDF data dump that is available from:

[http://download.geonames.org/export/dump/all-geonames-rdf.txt.zip]

is a little idiosyncratic. Rather than provide a single ntriples or even RDF/XML file 
the dump consists of a text file that consists of alternating lines like this:

  ...feature URI....
  <rdf:RDF>...RDF/XML description of feature....</rdf:RDF>

This means you need to script up unpacking the file in order to load it into a triple 
store.

The Rakefile and script provided here make it easy to download, unpack and convert 
the data dump into ntriples.

The conversion is written in Ruby and uses the RDF.rb and rdf/raptor libraries.
So before running the scripts you'll need to execute:

  sudo gem install rdf rdf-raptor

To download and convert the files into ntriples simply run:

  rake download convert

The converted data is stored in geonames.nt.

License
-------

These scripts are free and unencumbered public domain software. For more information, see http://unlicense.org/ or the accompanying UNLICENSE file.
