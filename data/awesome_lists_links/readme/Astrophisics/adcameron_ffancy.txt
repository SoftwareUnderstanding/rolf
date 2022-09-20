# ffancy

FFANCY: A pulsar searching software package designed for testing of and experimentation with the Fast Folding Algorithm (FFA)
Developed by Andrew Cameron, MPIfR IMPRS PhD research student, in collaboration with Ewan Barr & David Champion

Contact: acameron@mpifr-bonn.mpg.de

1. Installation

   FFANCY is self contained, and requires only standard external C libraries for installation. Simply run 'make all' and the full set of executables will be compiled in the root directory.

2. Usage

   The programs included in this package serve the following functions:

   * ffancy: runs an implementation of the FFA on real or simulated pulsar time series data in either SIGPROC or PRETSO format  with a choice of additional algorithms to be used in the evaluation of each folded profile. Outputs a periodogram along with other output threads used for testing purposes.
   * progeny: generates simulated pulsar profiles for use in testing profile evaluation algorithms independent of the FFA. Allows for multiple profile components and shapes including pulse scattering.
   * prostat: provides basic statistics for the folded profiles produced by progeny
   * metrictester: allows for testing of the individual profile evaluation algorithms independent of the FFA, using profiles produced by progeny
   * add_periodograms: adds two periodograms together. Experimental program, treat results with caution
   * ffa2best: converts the periodogram output from ffancy into a list of pulsar candidates, with options for candidate grouping and harmonic matching

   Running './program -h/--help' will provide detailed help and usage instructions for each individual program in this suite

3. Terminology changes

   During the development of this software, the profile evaluation algorithms were previously referred to as "metrics" and used a modified numbering system. The primary Algorithms, 1 & 2, were previously known as Metrics 6 & 7 respectively. Secondary Algorithms 3 - 8 were previously known as Metrics 0 - 5 respectively. Although the public interfaces for these programs reflect the new terminology, please bear in mind that the source code may still reference the old terminology conventions. 