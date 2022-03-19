# HiBISCuS
HiBISCuS: Hypergraph-Based Source Selection for SPARQL Endpoint Federation

## Source Code
The HiBISCuS source code along with all of the 3 extensions (SPLENDID, FedX, DARQ) can be checkout from project website https://github.com/AKSW/HiBISCuS/

## FedBench
FedBench queries can be downloaded from project website https://code.google.com/p/fbench/

## Datasets Availability
All the datasets and corresponding virtuoso SPARQL endpoints can be downloaded from the project old website https://code.google.com/p/hibiscusfederation/. 


## Usage Information
In the following we explain how one can setup the BigRDFBench evaluation framework and measure the performance of the federation engine.

## SPARQL Endpoints Setup
The first step is to download the SPARQL endpoints (portable Virtuoso SAPRQL endpoints from second table above) on different machines, i.e., computers. Best would be one SPARQL endpoint per machine. Therefore, you need a total of 13 machines. However, you can start more than one SPARQL endpoints per machine.
The next step is to start the SPARQL endpoint from bin/start.bat (for windows) or bin/start_virtuoso.sh (for Linux). Make a list of the all SPARQL endpoints URL's ( required as input for index-free SPARQL query federation engines, i.e., FedX). It is important to note that index-assisted federation engines (e.g., SPLENDID, DARQ, ANAPSID) usually stores the endpoint URL's in its index. The local SPARQL endpoints URL's are given above in second table.
Running SPARQL Queries
Provides the list of SPARQL endpoints URL's, and a FedBench? query to the underlying federation engine. The query evaluation start-up files for the selected systems (which you can checkout from project website https://github.com/AKSW/HiBISCuS/) are given below.

----------FedX-original-----------------------

package : package org.aksw.simba.start;

File:QueryEvaluation?.java

----------FedX-HiBISCuS-----------------------

package : package org.aksw.simba.fedsum.startup;

File:QueryEvaluation?.java

----------SPLENDID-original-----------------------

package : package de.uni_koblenz.west.evaluation;

File:QueryProcessingEval?.java

----------SPLENDID-HiBISCuS-----------------------

package : package de.uni_koblenz.west.evaluation;

File:QueryProcessingEval?.java

----------ANAPSID-----------------------

Follow the instructions given at https://github.com/anapsid/anapsid to configure the system and then use anapsid/ivan-scripts/runQuery.sh to run a query.
