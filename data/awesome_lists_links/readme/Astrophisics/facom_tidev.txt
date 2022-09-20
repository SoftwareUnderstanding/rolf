tidev
=====

A general framework to calculate the evolution of rotation for tidally
interacting bodies using the formalism by Efroimsky.

Quick start
-----------

1. Get a copy of tidev from https://github.com/facom/tidev:

   NOTE: You can get an anonymous clone of the project using:

       $ git clone git://github.com/facom/tidev.git

   Then you will be able to get updates using 'git pull'.

2. Install dependencies:

   You can install dependencies in the system:

       $ sudo apt-get install libgsl0-dev libgsl0ldbl

       $ sudo apt-get install libconfig++8 libconfig++-dev

   If dependencies installation does not work properly you can
   download sources and compile them directly into the util directory.

   Sources for gsl: http://www.gnu.org/software/gsl
   
   Sources for libconfig: http://www.hyperrealm.com/libconfig

   You should be sure that binary libraries and header files from both
   dependencies are properly placed into the util/include and util/lib
   directories.

   Latest realease of the package include sources for gsl and
   libconfig.  To install them and compile the package follow the
   procedure:

   a. Untar sources:

      $ cd util/src

      $ tar zxvf <gsl_sources>.tgz

      $ tar zxvf <libconfig_sources>.tgz

   b. Configure, compile and install (locally):

      $ cd <package>

      $ ./configure --prefix=$(pwd) && make && make install

      $ cd ..

      Where <package> is each of the installed dependencies (gsl and libconfig)

   c. Copy library and header to local directories:

      $ cd <package>

      $ cp -rf lib/* ../../lib

      $ cp -rf include/* ../../include

3. Configure system:

       $ nano tidev.cfg

4. Compile:

       $ make tidev-resonances.out

   If you are using GSL and Libconfig versions compiled from the
   sources in the util directory make using makefile.local instead the
   default makefile:
   
       $ make -f makefile.local tidev-resonances.out

4. Run:

       $ ./tidev-resonances.out

To know more read the MANUAL.md(html).

For the contirbutor
-------------------

1. Generate a public key of your account at the server where you will
   develop contributions:

   $ ssh-keygen -t rsa -C "user@email"

2. Upload public key to the github project site
   (https://github.com/facom/tidev).  You will need access to the
   account where the repository was created.

3. Configure git:

   $ git config --global user.name "Your Name"
   $ git config --global user.email "your@email"

4. Get an authorized clone of the master trunk:

   $ git clone git@github.com:facom/tidev.git

License
-------

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301, USA.

Copyright (C) 2013 Jorge I. Zuluaga, Mario Melita, Pablo Cuartas,
Bayron Portilla

---------------------

This file has been format using
[Markdown](http://daringfireball.net/projects/markdown).