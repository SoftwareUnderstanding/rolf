xodx
====

This is an implementation of the basic functionalities of a DSSN Provider:
* [Semantic Pingback](http://aksw.org/Projects/SemanticPingback) for Friending
* [Pubsubhubbub](http://code.google.com/p/pubsubhubbub/) (PuSH) for notification along the edges

It is written in PHP and utilizes the Zend Framework and the [Erfurt Framework](http://erfurt-framework.org/)

Installation
------------
You need a webserver (tested with Apache, but I hope it also runs with nginx and lighttd) and a database backend which is supported by Erfurt (MySQL and Virtuoso).

### Erfurt
Run `git submodules init` and `git submodules update` to clone Erfurt.

Take one of the prepared `config.ini-*` files in `xodx/libraries/Erfurt/library/Erfurt`, copy it to `config.ini` and configure it according to your system setup.

### Zend
You have to place a copy of the Zend framework library into `libraries/Zend/` you can do this by doing the following things (replace `${ZENDVERSION}` e.g. with `1.12.0`):

    wget http://packages.zendframework.com/releases/ZendFramework-${ZENDVERSION}/ZendFramework-${ZENDVERSION}-minimal.tar.gz
    tar xzf ZendFramework-${ZENDVERSION}-minimal.tar.gz
    mv ZendFramework-${ZENDVERSION}-minimal/library/Zend libraries
    rm -rf ZendFramework-${ZENDVERSION}-minimal.tar.gz ZendFramework-${ZENDVERSION}-minimal

### JavaScript
You have to add [twitter bootstrap](http://twitter.github.com/bootstrap/) and [jquery](http://jquery.com/) to the `resources` directory.

Code Conventions
----------------
Currently, this project is developed using [OntoWiki's coding standard](http://code.google.com/p/ontowiki/wiki/CodingStandard).
