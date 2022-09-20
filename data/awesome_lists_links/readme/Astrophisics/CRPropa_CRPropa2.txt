
CRPropa 2.0
========

### Installation

You need a g++ version > 3 and the associated g77
You need ~750 Mo of memory for the installation

1) You must have CLHEP 2.0.4.3 and CFITSIO 3.006 installed
	 If they are not installed, you should either intall them
	 yourself or run get_external.sh which will automatically
	 download and install in the proper way those 2 package in the
	 External directory.
         * NOTE : You need gcc / g++ and some common tools to run
	 properly the get_external.sh shell script.
	 * WARNING : You need 700 Mo of disk space mostly due
         to the size of the CLHEP library. Compiling this library
         takes a bit of time.

   1. Remark: 	Contrarily to the v1.0 release, you do not need to set the
              	environment variables CLHEP_DIR and CFITSIO_DIR.
   2. Remark: 	If you want to use your own version of CLHEP or CFITSIO,
		make sure to set the pathes to the correspnding directories
		during your configure call later, e.g. (s. below):
			./configure --prefix=$YourInstallationPath$
			--with-clhep-path=$YourCLHEPPath/bin/
			--with-cfitsio-include=$YourCFITISO/include
			--with-cfitsio-library=$YourCFITISO/lib
   3. Remark:	CLHEP is a shared library. Thus, its /lib path should be
		include in your LD_LIBRARY_PATH variable.

2) Since CRPropa version 1.5 a root installation is needed. Hence,
 	install root and set the proper pathes.

   1. Remark:	By default CRPropa will use the root version you have
		properly installed on your system (use 'which root' in your
		shell to find out which version this is). You can compile
		CRPropa using another root version using an additional flag
		while calling the configure script later:
			--with-root=$YourROOTPath/lib/
		Currently we recommend to use root version 5.30.
   2. Remarks:	The executable root-config should exist in your root's /bin directory.
		But, sometimes there is only a file called root-configX.Y where
		X.Y is the root version number. In this case you can create a symbolic
		link called root-config which is simply calling root-configX.Y.

3) Rebuild the Makefile by calling:
    	    autoreconf -ivf

4) Run ./configure in the trunk/ directory. In particular you
    can use the following options:
       --prefix=PREFIX     where the executable, libraries and various
            tables will be put after "make install". Default: /usr/local/
       --with-cfitsio-include=DIR      cfitsio library include files
            are in DIR. Default: trunk/External/cfitsio/include (as
	    set when using get_external.sh)
       --with-cfitsio-library=DIR      cfitsio library file is in DIR.
            Default: trunk/External/cfitsio/lib (as set when
	    using get_external.sh)
       --with-clhep-path=DIR      clhep-config binary is in DIR.
            Default: trunk/External/bin (as set when using
	    get_external.sh)

  1. Remark:	CRPropa is currently not compatible with gfortran 4.6. An older version should
		be used, gfortran 4.4 is known to work. In case gcc 4.6 is used, setting
		LDFLAGS="-Wl,--no-as-needed" is required.

		For Ubuntu 11.10 the following configure options have to be used (+install gfortran-4.4)
		./configure CXXFLAGS="-g -O2 -DUBUNTU" LDFLAGS="-Wl,--no-as-needed" F77=gfortran-4.4

5) Start the bash script in the trunk/ folder to download and install the photo disintegration
    data package via:
    	 ./GetPDCrossSections.sh

6) Run "make" in the trunk/ directory. This will run the Makefiles
    in the various subdirectories.
	* NOTE : If your machine has more than one processor, you should
	for example run "make -j 4" to make quickly the CRPropa package,
	e.g. here on 4 processors.

7) Run "make install" in the trunk/ directory.

8) Tests: go to $(datadir)/examples/GettingStarted and call
      $(bindir)/CRPropa source1d.xml ...
    You can then play with the various configuration files (source*d.xml, traj*d.xml).

### Contents of the tar ball

examples/GettingStarted/: simple XML configuration files and associated
density and magnetic field files to run tests
Note that the magnetic grid that is given (smallB.fits) is
a simple subgrid to test the code, and in particular it
does not have periodic boundary conditions.

External: external packages, ie. TinyXML

dint: the DINT package (written in C)
sophia: the SOPHIA package (written in Fortran)
src: the CRPropa package

doc: the user guide and Doxyfile

### Documentation

The CRPropa2.0 manual is available in doc/UserGuide.pdf.

A doxygen documentation cam be generated in the doc/html subdirectory by "make" if you have doxygen installed. We recommend doxygen-1.7.6.1 or later versions.

Command line help with the man command is available if you set
  MANPATH=$MANPATH:$(mandir)
where $(mandir) is $(prefix)/man by default.


