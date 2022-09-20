# BiPoS1
This is the first version of the Binary Population Synthesizer (BiPoS1). It allows to efficiently calculate binary distribution functions after the dynamical processing of a realistic population of binary stars during the first few Myr in the hosting embedded star cluster. Instead of time-consuming N-body simulations, BiPoS1 uses the stellar dynamical operator, which determines the fraction of surviving binaries depending on the binding energy of the binaries. The stellar dynamical operator depends on the initial star cluster density, as well as the time until the residual gas of the star cluster is expelled.  At the time of gas expulsion, the dynamical processing of the binary population is assumed to effectively end due to the expansion of the star cluster related to that event. BiPoS1 has also a galactic-field mode, in order to synthesize the stellar population of a whole galaxy.

# setup and use

This is a short version of how to setup and use BiPoS1; for a more detailed version see the paper by Dabringhausen, Marks and Kroupa (2021, coming soon...)

When extracting the program files of BiPoS1 to a computer, the user should take care that the file structure is preserved: The folders "Lib" and "output" are essential for the storage of the output data of BiPoS1. BiPoS1 is compiled by typing "make BiPoS" into the command line of a terminal at the directory where the user has stored the program. BiPoS1 should start afterwards, if "./BiPoS" is typed into the command line.

After typing ./BiPoS, the user is directed to the help menu. The user can choose here one of four commands, which explain the syntax used in BiPoS1 further. These are:

- ./BiPoS genlib help: Instructions for how to create a library of binary systems, which is used in later calls of BiPoS1. First-time users have to generate a library of binaries first!
- ./BiPoS clust help: Instructions for how to synthezise a population of binaries in a single star cluster.
- ./BiPoS field help: Instructions for how to synthesize a galaxy-wide field population.
- ./BiPoS SpT help: This command shows the default mass ranges for spectral types and a way to change them.

Proficient users may skip the help menu, and type in directly the syntax for the binary population they want BiPoS1 to sythesize. An example for a working command would be "./BiPoS clust mecl=500 rh=0.5 t=1 SpT=M". It creates the surviving binary population of M-stars after 1 Myr in a star cluster with an embedded mass of 500 solar masses and a half-mass radius of 0.5 pc. There are more examples like this in the help menu, besides a more thorough guide how to use BiPoS1.

Have fun!
