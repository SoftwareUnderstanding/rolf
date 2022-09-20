
# HDS re-implemented on top of HDF5

The Starlink Hierarchical Data System is a hierarchical data format
initially developed in 1982. It is the data format that underlies the
4 million lines of code in the Starlink software collection. HDS was
never adopted outside of the Starlink community and associated
telescopes and is now a niche data format. HDS has undergone numerous
updates over the years as it was rewritten in C from BLISS and ported
from VMS to Unix. It works well but there is no longer any person
capable of supporting it or willing to learn the details of the
internals. HDS is also an impediment to wider adoption of the
N-dimensional Data Format (NDF) data model.

Despite being developed in the late 1980s, HDF (and currently HDF5)
has been adopted much more widely in the scientific community and is
now a de facto standard for storing scientific data in a
hierarchical manner.

## libhdsh5

This library is an attempt to reimplement the HDS API in terms of the
HDF5 API. The idea is that this library could be dropped in as a
direct replacement to HDS and allow the entire Starlink software
collection to run, using HDF5 files behind the scenes which can then
be read by other general purpose tools such as h5py.

## Migration

I'm not yet worrying about migration of HDS v4 files to HDF5. Should
there be a wrapper HDS library that manages to forward calls to either
the native HDS or HDS/HDF5 based on the file that is being opened? Or
should we just write a conversion tool to copy structures from HDS
classic to HDF5?

## Authors

Tim Jenness
Copyright (C) Cornell University.

BSD 3 clause license.

Currently some high level HDS files are included that use a GPL
license and the original CLRC copyright. These implement functionality
using public HDS API and so work with no modifications. Their
continued use will cause some rethinking on license but that's for a
later time.

## Porting notes

### Error handling

HDF5 maintains its own error stack and HDS (and Starlink) uses EMS. All calls to HDF5
are wrapped by an EMS status check and on error the HDF5 stack is read and stored
in EMS error stack.

### datFind

datFind does not know whether the component to find is a structure or primitive
so it must query the type.

### datName

H5Iget_name returns full path so HDS layer needs to look at string to
extract lowest level.

### TRUE and FALSE is used in the HDF5 APIs

but not defined in HDF5 include files.

### Dimensions

HDS uses hdsdim type which currently is an 32 bit int. HDF5
uses hsize_t which is an unsigned 64bit int. In theory HDS
could switch to that type in the C interface and this will
work so long as people properly use hdsdim.

More critically, HDF5 uses C order for dimensions and
HDS uses Fortran order (even in the C layer). HDF5
transposes the dimensions in the Fortran interface to
HDF5 and we have to do the same in HDS.

### Memory mapping

Not supported by HDF5. Presumably because of things like
compression filters and data type conversion.

datMap must therefore mmap and anonymous memory area to
receive the data and then datUnmap must copy the data back
to the underlying dataset. Must also be careful to ensure
that `datGet`/`datPut` can notice that the locator is memory
mapped and at minimum must complain.

Will require that `datMap` and `datUnmap` lose the `const` in
the API as HDSLoc will need to be updated. Also need `datAnnul`
to automatically do the copy.

Must work out what happens if the program ends before the
datUnmap call has been done and also how `hdsClose` (or
`datAnnul` equivalent on the file root) can close all the mmapped
arrays.

What happens if the primitive is mapped twice? With different
locators? What happens in HDS?

### H5LTfind_dataset

Strange name because it also finds groups.

### _LOGICAL

In HDS (presumably Fortran) a `_LOGICAL` type seems to be
a 4 byte int. This seems remarkably wasteful so in this
library I am using a 1 byte bitfield type.

I have changed the C interface to use hdsbool_t boolean type
so as to be explicit about the types.

Currently hdsbool_t is a 32-bit integer. Internally a 1 byte type is used.
and externally a 4 byte type is used. This means that the routines
that query the HDF5 file for type have to do an additional check
to see what the in memory type is meant to be.

### datLen vs datPrec??

How do these routines differ?

SG/4 says Primitive precision vs Storage precision

`datPrec` doesn't seem to be used in any of Starlink.

`datLen` is called but in some cases seemingly as an alias
for `datClen`.

### Chunking and dataset creation

HDS allows primitives to be reshaped but HDF5 only allows that
if you have specified the maximum size in advance. For now
H5S_UNLIMITED is used as the maximum size but this requires that
we define chunking. In the absence of any ability for the HDS API
to provide guidance the library currently chunks based on the size
of the array being created.

### datRenam

Need to test whether `datRenam` or `datMove` break locators
that are previously associated with the objects being moved/renamed.
H5Lmove indicates that they should still work fine.

### datWhere

... is probably not going to work (and is only visible in the Fortran
API).

### datSlice / datCell

`datSlice` and `datCell` (for primitive types) are both attempting to
select a subset of a dataspace.

The `datasapce_id` element in the `HDSLoc` must be treated as the requested
dataspace by the user and not be confused by the dataspace associated with
the primitive datset itself. `datVec` has a similar issue in that the
locator from a `datVec` call is must to be rank 1 regardless of the underlying
dataset and if queried by `datShape` it should return the vectorized dimensions.
Currently we do not change the dataspace for datVec and datShape is short-circuited
to understand the vectorized special case.

HDS only supports the concept of 'sub-setting' and does not support
sampling or scatter-gather.

Slicing of a vectorized data structure requires extra work because a contiguous
subset of the N-D array has to be selected. Currently this is done by selecting points
from the dataspace.

### datPrmry

Not sure if this is possible in HDF5. HDS has a distinction between a primary
and secondary locator and annulling a secondary locator will not affect
whether the file is kept open or not. If HDF5 does not support this then
we can simply try to assert that all locators are primary locators, or else
try to keep track of this internally in the wrapper.

### Incompatibilies

HDF5 does not support an array of structures. Structures must be supported
by adding an explicit collection group. For example:

```
   HISTORY        <HISTORY>       {structure}
      CREATED        <_CHAR*24>      '2009-DEC-14 05:39:05.539'
      CURRENT_RECORD  <_INTEGER>     1
      RECORDS(10)    <HIST_REC>      {array of structures}

      Contents of RECORDS(1)
         DATE           <_CHAR*24>      '2009-DEC-14 05:39:05.579
```
will have to be done as

```
  /HISTORY
    Attr:TYPE=HISTORY
      /CREATED
      /CURRENT_RECORD
      /RECORDS
        Attr:TYPE=HIST_REC
        Attr:ISARRAY=1
        HDSCELL(1)
           Attr:TYPE=HIST_REC
           /DATE
        HDSCELL(2)
           Attr:TYPE=HIST_REC
           /DATE
```

We name the cell with its coordinates included to make it easy for
datCell to select the correct group simply by converting the
coordinates to (a,b). It also means that hdsTrace can easily be
modified to handle the presence of the group in the hierarchy.
If the name looks like /HISTORY/RECORDS/HDSCELL(5) hdsTrace
can easily convert that to the syntactically correct
.HISTORY.RECORDS(5) by removing all occurrences of "/HDSCELL".

Note also that in the final system HDSCELL is replaced with a longer
string that we know is longer that DAT__SZNAM (so that we can not
end up with the name by mistake or fluke through the HDS layer).

Slices of an array structure break this, as does vectorization (which
is used by HDSTRACE a lot).

### Type conversion

HDS can do type conversion from/to strings for numeric and logical
types. HDF5 can not do that in a H5Dwrite call so the type conversion
has to happen in datPut before that. The parameter system relies on this
as parameters are always supplied as strings but the correct values
are stored in the HDS parameter file.

## HDS Groups

`hdsLink` and `hdsFlush` are implemented above HDF5. A hash table and
vector of locators is used to keep track of group membership. Uses
uthash.

## hdsClose

`hdsClose` is documented to be obsolete but it is still called throughout
ADAM. Currently just calls `datAnnul`.

## Primary locators?

Does HDF5 have the concept of a secondary locator?

HDS defaults to top-level locators being "primary" and all other locators being
"secondary". HDS is allowed to close the file if all primary locators are annulled.
A secondary locator can be promoted to a primary locator using `datPrmry`. In
HDF5 it seems that all locators are primary.

In Starlink software only one routine in NDF every sets a primary locator
to a secondary locator (`ndf1_htop.f`), all others are setting to primary.

## hdsLock/hdsFree

`hdsLock` requires access to the underlying file descriptor associated with
the file. This is hard to obtain and no Starlink software calls the routine.
`hdsFree` will not complain as it is only meant to do something if `hdsLock`
has been called.


## TO DO

`datRef` and `datMsg` currently do not report the full path to a file,
unlike the HDS library. This is presumably an error in `hdsTrace()`
implementation which may have to determine the directory.  (and
furthermore we may have to store the directory at open time in case a
`chdir` happens).

`datParen` does not currently work out when it hits the top of the
group hierarchy.
