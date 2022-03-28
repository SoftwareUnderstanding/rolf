# QOSE (Quantum Optimal Subarchitecture Extraction)
_pronounciation: "cozy"_ 

#### Contributors:
* Jelena Mackeprang 
* Roeland Wiersema
* Aroosa Ijaz
* Kathleen Hamilton
* Yash Chitgopekar

Determining the optimal quantum circuit structure for a parameterized model that will be trained as a quantum circuit classifier.  Heuristic ansatz design involves searching over a large parameter space including: depth, number of qubits, entangler layer design, initialization, optimizer choice (and associated hyperparameters), embedding..

Since running exponentially large circuits are liable to run into barren plateau problems, there is a need for fast searching over circuit structures coupled with intialization choices.

QOSE is a version of OSE (optimal subarchitecture extraction) that is an adaptation of the FPTAS algorithm introduced in arxiv.org/abs/2010.08512 for parameterized quantum circuits and written using PennyLane. 

Documentation can be found [here](https://kmz4.github.io/QHACK2021/).

### Example notebooks:
For basic functionality of QOSE
[QOSE: Finding the optimal circuit architecture](demos/QOSE_demo.ipynb)

For exploring the effects of different embedding methods
[QOSE: Different Embeddings](demos/QOSE_different_embeddings.ipynb)

For a demonstration of parallelized execution 
[QOSE: MPI and SV1 parallelization on AWS](demos/QOSE_demo_parallel.ipynb)


## Installation instruction 

```
git clone <this repo>
cd QHACK2021
python setup.py install
```

### References

    De Wynter (2020) "An Approximation Algorithm for Optimal Subarchitecture Extraction" 
    [arxiv.org/abs/2010.08512]
    De Wynter and Perry (2020) "Optimal Subarchitecture Extraction for BERT" 
    [arxiv.org/abs/2010.10499]

_this module was developed as part of QHACK 2021_

