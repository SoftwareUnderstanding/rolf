# rcsvw

Package that implements the candidate recommendations from the W3C CSV on the Web Working Group.

# Copyright / License

## rcsvw package

Copyright (C) 2011-2015  Davide Ceolin

## Authors / Contributors

Author: Davide Ceolin

This package relies on an extended version of the rrdf package. The instructions below allow installing this version of rrdf, before installing csvw.

# Install from R

    > install.packages("rJava") # if not present already
    > install.packages("devtools") # if not present already
    > library(devtools)
    > install_github("davideceolin/rrdf", ref="extended", subdir="rrdflibs")
    > install_github("davideceolin/rrdf", ref="extended", subdir="rrdf", build_vignettes = FALSE)
    > install_github("davideceolin/rcsvw")
