## SparsePz

### **Sparse Representation of Photometric Redshift PDFs**

This package will soon implemented as part as the [MLZ](https://github.com/mgckind/MLZ) package repository. This is the standalone version. The full documentation of MLZ is located [here](http://lcdm.astro.illinois.edu/static/code/mlz/MLZ-1.0/doc/html/index.html)

For more information and a implementation of this technique, check this [paper](http://arxiv.org/abs/1404.6442) 

Requirements:

* scipy
* matplotlib
* numpy
* pyfits (to store output file) >= 3.3
* mpi4py (optional for parallel running)


To run:

    python example_sparse.py

To check the results:

    python read_sparse.py


The format of the original PDF file is given in a numpy array but can eb easily change and corresponds to a 2D array where each row is the PDF and the very last row are the redshift positions.

