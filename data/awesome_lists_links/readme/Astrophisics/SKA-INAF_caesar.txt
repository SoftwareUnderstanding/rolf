<p align="left">
  <img src="share/CaesarLogo.png" alt="Caesar source finder logo"/>
</p>

[![Build Status](http://jenkins.oact.inaf.it:8080/buildStatus/icon?job=SKA/CAESAR)](http://jenkins.oact.inaf.it:8080/me/my-views/view/All/job/SKA/job/CAESAR/)

# CAESAR

Compact And Extended Source Automated Recognition

## **About**  
CAESAR is a C++ software tool for automated source finding in astronomical maps. It is distributed for research use only under the GNU General Public License v3.0. 

## **Credit**
If you use CAESAR for your research, please acknowledge it in your papers by citing the following paper:

* `S. Riggi et al., "Automated detection of extended sources in radio maps:
progress from the SCORPIO survey", MNRAS (2016) doi: 10.1093/mnras/stw982, arXiv:1605.01852`
* `S. Riggi, F. Vitello et al., "CAESAR source finder: recent developments and testing", submitted to PASA (2019)`

or consider including me (`S. Riggi, INAF - Osservatorio Astrofisico di Catania, Via S. Sofia 78, I-95123, Catania, Italy`)
as a co-author on your publications.

## **Installation**  

### **Prerequisites**
Install the project mandatory dependencies:  

* ROOT [https://root.cern.ch/], to be built with FITSIO, PyROOT, RInterface options enabled. Make sure that the FindROOT.cmake is present in $ROOTSYS/etc/cmake directory after installation.
* OpenCV [http://opencv.org/]
* log4cxx [https://logging.apache.org/log4cxx/]
* boost [http://www.boost.org/] 
* cfitsio [https://heasarc.gsfc.nasa.gov/fitsio/fitsio.html], to be built with multithread support (e.g. give --enable-reentrant option in configure)
* protobuf [https://github.com/google/protobuf]
* jsoncpp [https://github.com/open-source-parsers/jsoncpp]
* cmake (>=2.8) [https://cmake.org]  
  
Optional dependencies are:
* MPICH [https://www.mpich.org/] or OpenMPI [https://www.open-mpi.org/], needed when the build option ENABLE_MPI=ON (to enable parallel source finding application)       
* OpenMP [http://www.openmp.org/], needed when the build option BUILD_WITH_OPENMP=ON (to enable multithread processing)  
* R [https://www.r-project.org/] and additional packages: RInside, Rcpp, rrcovHD, truncnorm, FNN, akima. Needed when the build option ENABLE_R=ON
* GoogleTest [https://github.com/google/googletest], needed for unit testing when the build option ENABLE_TEST=ON   
* Doxygen [www.doxygen.org/], needed to generate the API documentation   
* Sphinx [http://www.sphinx-doc.org] & Breathe [https://pypi.org/project/breathe], needed to generate the Sphinx API & wiki documentation

Dependencies for the provided scripts are:
* python (>=2.7) [https://www.python.org/] and these additional modules: pyfits, astropy, scipy, getopt, argparse, collections, matplotlib, pylab
* CASA [https://casa.nrao.edu/]


Make sure you have set the following environment variables to the external library installation dirs 
* ROOTSYS
* OPENCV_DIR
* BOOST_ROOT
* LOG4CXX_ROOT
* JSONCPP_ROOT
* R_DIR (optional)

Add also the following paths to the PKG_CONFIG_PATH environment var: 
* $LOG4CXX_ROOT/lib/pkgconfig  
* $JSONCPP_ROOT/lib/pkgconfig

CAESAR depends also on the wcstools and linterp libraries which are already provided in the external/ directory. 
**NB: The provided wcslib was slightly modified with respect to the original release to avoid naming conflicts with the R package due to some #define macros used in WCS.**

cmake should find all needed include dirs and libraries used to build the project.

### **Build and install**
To build and install the project:

* Clone this repository into your local $SOURCE_DIR  
  ```git clone https://github.com/SKA-INAF/caesar.git $SOURCE_DIR```
* Create the build and install directories: $BUILD_DIR, $INSTALL_DIR  
* In the build directory:  
  ```cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DENABLE_TEST=ON -DBUILD_WITH_OPENMP=ON -DENABLE_MPI=ON -DBUILD_APPS=ON $SOURCE_DIR```   
  ```make```  
  ```make install```  
  
### **Documentation generation**
To generate and install the API documentation you must have Doxygen (+dot) installed on your system. Enter the build directory and type:
  ```make doc```  

To generate and install the Sphinx API and wiki documentation you must have Sphinx + Breathe installed on your system. Enter the build directory and type:
  ```make doc-sphinx``` 
  
Online documentation is available at: https://caesar-doc.readthedocs.io/en/latest/  
  
### **Running unit tests**
To build the unit tests you must have Google Test installed and the ENABLE_TEST option set to ON when building Caesar.
To run the test:

```make test```

or alternatively run the script `runUnitTests` installed in the Caesar installation dir.
