# MeSsI Algorithm v3.0

This package contain all the neccesary functions to perform the automatic analysis and classification of merging clusters using the method developed in (https://arxiv.org/abs/1509.02524) by Martín de los Rios, Mariano Domínguez, Dante Paz & Manuel Merchán.

# Prerequisites

This software make an extensive usage of the following packages that must be installed: ```randomForest```, ```nortest```, ```cosmoFns```, ```mclust```, ```e1071```, ```beepr```, ```caret```, ```progress```. 
You can install this packages inside an R-session with:

```R
install.packages(c('randomForest', 'nortest', 'cosmoFns', 'mclust', 'e1071', 'beepr', 'caret', 'progress'))
``` 

# Installation

You can install the package directly from your R session using the ```install_github``` function from the ```devtools``` package.

``` R
library('devtools')
install_github('MartindelosRios/MeSsI')
```

# Example

``` R
# Loading the MeSsI library.
library('MeSsI')

# Loading the data
data('GalaxiesDataset')

# Let's see the structure of this dataset
str(GalaxiesDataset)

# As you can see this dataset already have all the properties of the galaxies precomputed.
# We will remove this properties and start with a dataset with only the angular positions (ra, dec), 
#  the redshift (z), the identification of the cluster to which the galaxy belongs (id), the color (color)
#  and the r apparent magnitude (mag).

cat <- GalaxiesDataset[, (1:6)]
colnames(cat)[1] <- 'id'
str(cat)

# Then we just can apply the messi functions to this catalog, optionally given a name to the folder where all the 
#  outputs file will be saved.

messi(cat, folder = 'test')
```
You can find more examples at https://github.com/Martindelosrios/MeSsI/tree/master/Examples
# Authors

Martín de los Rios (ICTP-SAIFR/IFT-UNESP)
https://martindelosrios.netlify.com/ 

<div itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0003-2190-2196" href="https://orcid.org/0000-0003-2190-2196" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">https://orcid.org/0000-0003-2190-2196</a></div>
