#DREAM 
**Distributed RDF Engine with Adaptive Query Planner and Minimal Communication**

RDF and SPARQL query language are gaining wide popularity and acceptance. **DREAM** is a hybrid RDF system, which combines the advantages and averts the disadvantages of the centralized and distributed RDF schemes. In particular, DREAM avoids partitioning RDF datasets and reversely partitions SPARQL queries. By not partitioning datasets, DREAM offers a general paradigm for different types of pattern matching queries and entirely precludes intermediate data shuffling (only auxiliary data are shuffled). By partitioning only queries, DREAM suggests an adaptive scheme, which runs queries on different numbers of machines depending on their complexities. DREAM achieves these goals and significantly outperforms related systems via employing a novel graph-based, rule-oriented query planner and a new cost model.

DREAM is implemented in C and C++, and available as open-source under the MIT License.

Download DREAM
----------------------

You can download DREAM directly from the Github Repository. Github also offers a zip download of the repository if you do not have git.

The git command line for cloning the repository is:
```
git clone https://github.com/az-hasan/DREAM.git
cd DREAM
```


Building
------------------
The current version of DREAM was tested on Ubuntu Linux 64-bit 14.04. It requires a 64-bit operating system. 
 
 
Dependencies
------------------

DREAM has the following dependencies.

1. [g++ (>= 4.8)](https://gcc.gnu.org/gcc-4.8/)
2. [MPICH (>= 3.1)](https://www.mpich.org/downloads/)
3. [Boost](http://www.boost.org/)
4. [TBB](https://www.threadingbuildingblocks.org/) 

We use the [rdf3x-0.3.8](https://code.google.com/p/rdf3x/downloads/detail?name=rdf3x-0.3.8.zip&can=2&q=) engine as part of DREAM. We use the unpacked binaries [id2name and rdf3xquery](https://github.com/az-hasan/DREAM/wiki/Running-DREAM#rdf3x-binaries).

Usage 
----------------
The [Wiki entry](https://github.com/az-hasan/DREAM/wiki) provides a guide to install and run DREAM on a cluster.


Contributing
-------------------
1. Fork it ( https://github.com/[my-github-username]/DREAM/fork )
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request
