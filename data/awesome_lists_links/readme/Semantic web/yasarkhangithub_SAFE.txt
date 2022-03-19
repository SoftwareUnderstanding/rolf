# SAFE: SPARQL Federation over RDF Data Cubes with Access Control

SAFE, a SPARQL query federation engine that enables decentralised, access to clinical information represented as RDF data cubes with controlled access.

## Experimental Setup
The experimental setup (i.e. code, datasets, setting, queries) for evaluation of SAFE is described here.

### Code
The SAFE source code can be checkedout from [SAFE GitHub Page](https://github.com/yasarkhangithub/SAFE/). 

### Datasets
We use two groups of datasets exploring two different use cases, i.e. INTERNAL and EXTERNAL.

The first group of datasets **(INTERNAL)** are collected from the three clinical partners involved in our primary use case. These datasets contain aggregated clinical data represented as RDF data cubes and are privately owned/restricted.

The second group of datasets **(EXTERNAL)** are collected from legacy Linked Data containing sociopolitical and economical statistics (in the form of RDF data cubes) from the **World Bank**, **IMF (International Monetary Fund)**, **Eurostat**, **FAO (Food and Agriculture Organization of the United Nations)** and **TI (Transparency International)**.

External datasets used in evaluation experiments of SAFE can be downloaded from [SAFE External Datasets](https://goo.gl/6s4juv).

### Settings

Each dataset was loaded into a different SPARQL endpoint on separate physical machines. All experiments are carried out on a local network, so that network cost remains negligible. The machines used for experiments have a 2.60 GHz Core i7 processor, 8 GB of RAM and 500 GB hard disk running a 64-bit Windows 7 OS. Each dataset is hosted as a Virtuoso Open Source SPARQL endpoint hosted physically on separate machines. The details of the parameters used to configure Virtuoso are listed in Table 1 below. Virtuoso version 7.2.4.2 has been used in experiments.

**Table 1:** SPARQL Endpoints Configuration

| SPARQL Endpoint       | Port           | URL  | Virtuoso Config Parameters  |
| ------------- |-------------| -----| -----|
| IMF      | 8890 | {System-IP}:8890/sparql | NoB=680000, MDF=500000, MQM=8G |
| World Bank      | 8891      |   {System-IP}:8891/sparql | NoB=680000, MDF=500000, MQM=8G |
| TI | 8892      |    {System-IP}:8892/sparql | NoB=680000, MDF=500000, MQM=8G |
| Eurostat | 8893      |    {System-IP}:8893/sparql | NoB=680000, MDF=500000, MQM=8G |
| FAO | 8895      |    {System-IP}:8895/sparql | NoB=680000, MDF=500000, MQM=8G |

- *NoB = NumberOfBuffers*
- *MDF = MaxDirtyBuffers*
- *MQM = MaxQueryMem*

### Queries

A total of 15 queries are designed to evaluate and compare the query federation performance of SAFE against FedX, HiBISCuS and SPLENDID based on the metrics defined. We define ten queries for the federation of EXTERNAL datasets and five for the federation of INTERNAL datasets. Only ten queries (EXTERNAL dataset queries) are made available due to owner restrictions.

Queries used in evaluation experiments of SAFE can be downloaded from [SAFE Queries](https://goo.gl/WCCnx3). 

### Metrics

For each query type we measured (1) the number of sources selected; (2) the average source selection time; (3) the average query execution time; and (4) the number of ASK requests issued to sources.

## Team

[Yasar Khan](https://www.insight-centre.org/users/yasar-khan)

[Muhammad Saleem](http://aksw.org/MuhammadSaleem.html)

[Aidan Hogan](http://aidanhogan.com/)

[Muntazir Mehdi](https://www.insight-centre.org/users/muntazir-mehdi)

[Qaiser Mehmood](https://www.insight-centre.org/users/qaiser-mehmood)

[Dietrich Rebholz-Schuhmann](https://www.insight-centre.org/users/dietrich-rebholz-schuhmann)

[Ratnesh Sahay](https://www.insight-centre.org/users/ratnesh-sahay)
