# comb
This is the archive of source code for *comb*, the AT&amp;T Bell Labs singledish radio astronomy spectral line data reduction and analysis package. By the way, it is pronounced like the first syllable of "combine" (from whence the name derives - combining spectra), **not** like the hair styling implement!

# History
Comb was actively developed and maintained from the mid 1980s to ~2004 by Robert W. Wilson, Marc W. Pound, Antony A. Stark, and others.  Comb was originally written in Fortran and ported to C in the late 1980s mostly by use of f2c. comb had its own plotting library which supported most common graphics devices. As a grad student in the early 1990s, Pound modified the code to work natively with raw data from NRAO 12-m telescope (and installed it on-site), created general help and news commands, and developed a distribution system of annual releases with a self-install script, all of which significantly increased the uptake of the code.  At its peak, comb was installed on several architectures at dozens of institutions worldwide in support of data reduction from many singledish telescopes including Bell Labs 7-m, NRAO 12-m, DSN 70m, FCRAO 14-m, Arecibo, AST/RO, SEST, BIMA, STO.  It's most recent incarnation was for use with the Stratospheric Terahertz Observatory in 2011-12, where Chris Martin moved it into a (now-defunct) CVS repository and made changes to allow comb to work on OSX and link to cfitsio and X11 (partly done when he was at AST/RO).  A cookbook was written in 1990, available in the *doc* subdirectory.  The no-longer updated comb information page is at http://www.astro.umd.edu/~mpound/comb/comb.html

Comb has not been compiled by me recently on a modern 64-bit architecture. In 2011, I got the STO version working with gcc -m32.  I still have a working 32-bit binary from 2001 that I copy around as a change computers.   That will all end soon.

07/21/2011 -- The STO version is now in the branch "sto", recovered from a CVS checkout I had.  Among the STO changes:
  * make it work with xwindow displays
  * make it work on mac osx
  * compile on 64 bit architecture (with -m32)
  * link to CFITSIO library instead of the internal FITS routines
  * some new commands - gs, gi, lf
 
