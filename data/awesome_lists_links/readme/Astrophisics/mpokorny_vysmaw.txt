# vysmaw client library

**maw** /mȯ/
  _noun_
  * the jaws or throat of a voracious animal

**vys·maw** /'vizmȯ/
  _noun_
  * a library for receiving a fast stream of visbility data

The vysmaw client library is intended to facilitate the development of code for
processes to tap into the fast visibility stream on the National Radio Astronomy
Observatory's Very Large Array correlator back-end InfiniBand network. Please be
aware that this library, as well as the implementation of the fast visibility
stream at the VLA, is experimental in nature.

## Build dependencies

  * cmake, version 2.8 or later
  * gcc, tested on version 5.x and 6.1.0; other C compilers may work
  * libibverbs, OFED version 1.1.8 or later?
  * librdmacm, OFED version 1.1.8 or later?
  * glib-2.0, version 2.28 or later
  * Python, version 2.7 or later (including 3.x)
  * cython, version 0.24 or later

The above dependencies must be satisfied with "development" versions of
packages, where applicable.

The version numbers quoted above reflect those used in development so far; they
are subject to change, and may or may not correspond to strict version
requirements. If you successfully build this project, please send a note to the
repository owner with the versions of the above dependencies you used.

## Build instructions

Simple: run cmake, followed by make. Below are some cmake scripts that I've used
for development on two different systems to help get you started.

A debug build on a standard NRAO RHEL 6.6 machine, with a locally installed,
modern version of cmake, in a pyenv environment (Python v 2.7.11):

``` shell
GCC=/opt/local/compilers/gcc-6/bin/gcc
PYTHON_EXECUTABLE=$( python-config --prefix )/bin/python
PYTHON_LIBRARY=$( ls $( python-config --prefix )/lib/libpython*.so )
PYTHON_INCLUDE_DIR=$( ls -d $( python-config --prefix )/include/python* )

BUILD_DIR=./build

CMAKE="mkdir -p $BUILD_DIR && cd $BUILD_DIR && \
 ~/stow/cmake-3.6.3/bin/cmake -DCMAKE_BUILD_TYPE=Debug \
 -DCMAKE_C_COMPILER=$GCC -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
 -DPYTHON_LIBRARY=$PYTHON_LIBRARY -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR .."
echo $CMAKE
eval $CMAKE
```

A release build on a Ubuntu 16.10 machine:

``` shell
GCC=$( which gcc )
PYTHON_EXECUTABLE=$( which python3 )
PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so.1.0
PYTHON_INCLUDE_DIR=/usr/include/python3.5m

BUILD_DIR=./build

CMAKE="mkdir -p $BUILD_DIR && cd $BUILD_DIR && \
 cmake -DCMAKE_C_COMPILER=$GCC \
 -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE -DPYTHON_LIBRARY=$PYTHON_LIBRARY \
 -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR .."
echo $CMAKE
eval $CMAKE
```

I recommend building the project in a sub-directory of the top level project
directory. This allows one to keep the source tree clean, and easily remove all
build artifacts, including the cmake generated files.

``` shell
# from the top level project directory...
mkdir build
cd build
cmake .. # replace with your cmake command
```

## Build artifacts

Note that if you intend to build and run sample code from your own source tree,
you may have to set PYTHONPATH to point to the `py` sub-directory of the build
directory.

### vysmaw

The two primary artifacts produced by the build are a C language shared library,
and a Python extension with a Python/Cython interface to the shared library.

### vys

A smaller artifact is a _vys_ system configuration library, which supports both
_vysmaw_ and the visibility stream producers (_i.e._, the sending processes: the
correlator back-end or a simulator).

### vyssim

A distributed visibility stream simulator application. This application must be
launched (and is compiled) as an MPI application, using any MPI job launcher
that is compatible with the MPI library used to build the application. For the
time being, the only further usage instruction is available by starting the
application with the ```-h``` flag.

## Configuration files

Configuration files for both the _vysmaw_ and _vys_ libraries are available in
the source tree. These may be installed by the user, but are operationally
optional. As an alternative to, or in addition to, installing these files on a
system, they may be used as templates by application developers for
application-specific configurations.

## Installation

To install the project libraries, include files and configuration files, a make
install target is provided. Note that the install directory may be set in the
call to cmake, using the `CMAKE_INSTALL_PREFIX` variable. An additional option
is to use the `DESTDIR` variable in the `make install` command, but keep in mind
that `DESTDIR` only sets a prefix to the value of `CMAKE_INSTALL_PREFIX` (which
defaults to `/usr/local`.)

As an example that illustrates the effect of the two variables used to define
the top-level installation directory, the following commands will result in
files installed under `/tmp/foo/vys/include`, `/tmp/foo/vys/lib`, and
`/tmp/foo/vys/etc`.

```shell
cmake -DCMAKE_INSTALL_PREFIX=/vys # ... truncated
make DESTDIR=/tmp/foo install
```

Finally, note that for a debug build, nothing will be installed when running
`make install`.

## Interfaces

### C API

At the moment, the API for the C language library is detailed in the `vysmaw.h`
source file. Python programmers are encouraged to refer the comments in that
file for documentation, as well.

### Python/Cython API

A complete API is available in Python, although production code should resort to
the Cython interface in some circumstances. In particular, for performance
reasons, the callback provided by the client, which is used to decide which
spectra are of interest to the client, ought to be implemented in Cython (or C).
The callback will be called very frequently by the client library under most
conditions; therefore, the callback is executed without locking the Python GIL,
and the callback itself should preferably **not** lock the GIL. Using an
inefficient callback will result in the client not receiving all the spectra
that it wishes to receive (although such a callback will not affect other
clients, or the correlator back-end.)

### Building a client application

To ensure binary compatibility for servers and clients, applications should be
compiled with the (gcc) "-fno-short-enums" flag. This advice applies to C, C++
and Cython applications.

### Running a client application

Efficient access to InfiniBand resources by vysmaw requires a large amount of
"locked" memory. However, many systems are configured to severely limit the
amount of locked memory available to user processes. While this limit may be
increased by using the `ulimit -l` command, there is often a hard limit on this
value that cannot be overridden by most users. A failure to access the required
amount of locked memory oftentimes appears as a failure to start an application,
with InfiniBand-related warning messages referring to "rdma" or "cq" (the exact
messages being dependent upon application error handling).

## Usage

Every client that initializes the library receives upon return from the
initialization function a vysmaw client handle, representing the resources
allocated by the library for that client, and a visibility data queue reference.
Upon initialization, a client provides the library with a callback function
predicate that is used by the library to determine the visibility spectra that
are to be delivered to the client _via_ the visibility data queue. Only those
spectra that satisfy the predicate will be passed to the client on its data
queue. After initialization, the client must simply take items (_i.e._, mostly
spectra) from the queue repeatedly, eventually call a shutdown function, and
continue to take items from the queue until a special, sentinel value is
retrieved. For efficiency in the library implementation, the memory used to
store spectra is a limited resource, which requires that client applications
make an effort to release references to spectral data as soon as
possible. Failure to release spectral data references in the client application
may result in failures of the client to receive all the spectra that are
expected.

## Sample code

All sample code can be found under the `examples` project directory.

### sample1 (mostly Python)

The [sample1](examples/sample1.pyx) application is trivial in that it uses a
callback that selects no spectra. It will run to completion on any machine, even
in the absence of an InfiniBand HCA. The application will simply print the
end-of-data-stream message to stdout, since no spectra are selected.

If there is no InfiniBand HCA, the library will immediately signal the end of
the data stream, and provide error messages to the client in the
end-of-data-stream message. Note that on some systems the OFS software may
insist on printing messages to stderr if no InifiniBand HCA is present.

Note that the sample uses the start_py method, which is convenient for
development and testing, but is not recommended for production code.

### sample2 (Python with Cython callback)

The [sample2](examples/sample2.pyx) application has the same functionality
as [sample1](examples/sample1.pyx), but, with a bit more usage of Cython and the
vysmaw Cython API (cy_vysmaw) than [sample1](examples/sample1.pyx), its
implementation avoids locking the Python GIL in the callback function predicate.

### sample3 (optimized Cython)

The [sample3](examples/sample3.pyx) application demonstrates several Cython
optimization techniques, as well as providing a non-trivial callback function
predicate. It implements a message processing loop in Cython that compiles
entirely to C, without entry to the Python interpreter.

### sample4 (C++)

[sample4](examples/sample4.cc) is an application written in C++. It can be used
as a simple, diagnostic vysmaw application, or as the basis for developing a
more interesting application. Starting the program with the `--timing` or `-t`
option may be useful for performance testing; starting without that option may be
preferable to review the metadata of the spectra that have been received.

### sample5 (C++)

[sample5](examples/sample5.cc) is similar to [sample4](examples/sample4.cc), but
writes received spectral data (along with metadata) to a binary file, or as text
to `stdout`. It may be useful for system test and diagnosis purposes.
