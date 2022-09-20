PIAO
====

PIAO (漂) is efficient memory-controlled python code using the standard SO algorithm to identify halos.


Reqiures (not fully checked!): 

    Python >= 2.6.x, not know for 3.x.x
    Numpy  >= 1.5.x
    Cython >= 0.18
    Mpi4Py >= 1.2
    Scipy  >= 0.8.x

To use:

    1, build ckdtree lib (SPH calculation included) with Cython by simply running build_lib : ./build_lib

    2. Now, I include the parameter file, change the parameters in this param.txt file or write your own.
	But, it need to be kept its format (similar to windows INI format, see python ConfigParser for more detail). 
	Use ./SO.py -h for more details.
    3. run the code by mpiexec -np 6 python SO.py param.txt

Note:

The ckdtree.pyx file is shamelessly stolen from 
https://github.com/scipy/scipy/blob/master/scipy/spatial/ckdtree.pyx.
Thanks to Patrick Varilly.
I only add SPH density calculation. Call Tree.qdens? for more information

Use the readgroups.py file to read the output of PIAO
