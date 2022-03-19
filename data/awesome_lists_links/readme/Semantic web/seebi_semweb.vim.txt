# vim bundle for working with RDF knowledge bases

## Installation

This bundle is created to work with `pathogen` so just copy the folder
into your bundle directory.

## Features

 * namespace completion
 * syntax highlightning
 * insert xsd:dateTime current time literal with abbrevation `xsdnow`

## namespace completion

You can use ctrl-x+u (autocomplete user function) for completing
rdf prefixes (as `owl`, `skos` and `foaf`) to namespace URIs (e.g.
`http://xmlns.com/foaf/0.1/` for the `foaf` prefix).

This done with the multifunctional shell script
[rdf.sh](https://github.com/seebi/rdf.sh) which is a dependency and must
be available as `rdf` in the path.

## syntax highlightning

### N3 Syntax

I just pull the syntax file from [Niklas
Lindström](git://github.com/vim-scripts/n3.vim.git) here.

### SPARQL Syntax

I just pull the syntax file from [Jeroen
Pulles](https://github.com/vim-scripts/sparql.vim) here.

