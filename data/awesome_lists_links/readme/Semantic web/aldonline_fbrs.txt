# FBRS = Facebook RDF Sync

Facebook's Graph API can return (almost) proper Linked Data when asked to in a polite manner. ( Accept: text/turtle ).
However, we can still not SPARQL the complete dataset.

Our solution is to load a subset of Facebook's graph at any given time.
Which subset to load is a tricky question and will depend on the use cases.
This framework supports some common ways of loading data from Facebook, including batch, incremental,
as well as keeping data in sync using Facebook's Real-Time updates.

## Setup

* You will need the following dependencies:
 * Node v0.6.x
 * OpenLink Virtuoso >= 6.1.3
 * Virtuoso's iSQL client library on the path

FBRS will access Virtuoso via both the iSQL client library and the SPARQL HTTP endpoint.
You can configure the ports later on.

Clone this repo:

    git@github.com:aldonline/fbrs.git

And install dependencies:

    cd fbrs
    npm install -d

For example:

* Load all data 
...

## Usage

You can run FBRS in batch mode or via an HTTP Server. Bath is simpler, but does not 
keep the data in sync.

### Batch ( load everything once or incremental )



### Web Server ( With Real-Time Updates )

    FBRS_SPARQL_ENDPOINT="http://localhost:8890/sparql"
    FBRS_VIRTUOSO_PORT=1111
    FBRS_VIRTUOSO_USERNAME="dba"
    FBRS_VIRTUOSO_PASSWORD="dba"
    FBRS_PORT=3008
    FBRS_CALLBACK_PORT=3009
    FBRS_ACCESS_TOKEN=...



