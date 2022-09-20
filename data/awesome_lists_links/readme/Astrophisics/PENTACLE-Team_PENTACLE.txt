PENTACLE
====

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

Overview
----

PENTACLE is a Parallelized Particle-Particle Particle-tree code for Planet formation, which is a parallelized hybrid N-body integrator executed on a CPU-based (super)computer. 

## Requirement

- GNU Compiler Collection
- MPI Library
- FDPS https://github.com/FDPS/FDPS

## Installation

1. Download PENTACLE code

        $ git clone https://github.com/PENTACLE-Team/PENTACLE.git
    
    * You also download FDPS library.

            $ git clone https://github.com/FDPS/FDPS.git
    
1. Describe the location of FDPS library in makefile

    You set the path to FDPS library at "PS_PATH".

1. Compile PENTACLE code

        $ make
