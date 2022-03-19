
# Sparql-cli

A command line SPARQL console. Works with endpoints as a well as local/remote graphs. 


### Status

Prototype. Runs but largely untested. 


### How-to

Suggested:

- Install via  `python setup.py install` then call `sparql-cli`


Otherwise, you can run in dev mode: 

- get to the source top folder and run `python -m sparql-cli.main -h`


### Summary

This is an attempt to build a sparql console using python-prompt toolkit

Idea:
either pass a sparql endpoint / or a graph URI which is loaded in memory using RDFLib
= useful to test things out quickly!

REQUIREMENTS
- click
- colorama
- rdflib
- pygments


@todo
- export results as html / web
- allow passing an endpoint @done
- add more saved queries 
- store endpoints? eg via an extra command line
- add meta level cli eg .show or .info etc..
- namespaces and shortened URIs


### Note 

This tool relies on pygments for the syntax highlighting, however the current Sparql Lexer included in Pygments stable releases is broken (https://bitbucket.org/birkenfeld/pygments-main/issues/1236/sparql-lexer-error) hence you either have to update it from the dev branch or wait for the 2.2 release of  Sparql Lexer.



### Changelog


**October 22, 2016**
- various improvements to print out
- added Endpoint connection and tested with DBpedia


**October 22, 2016**
- improved rendering of results 
- added options via click 


**October 21, 2016**
NOTE: > problem with Sparql Lexer
https://bitbucket.org/birkenfeld/pygments-main/issues/1236/sparql-lexer-error
"fixed in 2.2 release soon"

Improved by installing this commit
https://bitbucket.org/birkenfeld/pygments-main/commits/60afc531aa2b
>> installed manually 'hg clone https://bitbucket.org/birkenfeld/pygments-main'
and it works


