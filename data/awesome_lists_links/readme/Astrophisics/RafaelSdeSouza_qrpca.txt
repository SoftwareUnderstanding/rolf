![GitHub](https://img.shields.io/github/license/RafaelSdeSouza/qrprcomp) 
[![DOI](https://zenodo.org/badge/481248275.svg)](https://zenodo.org/badge/latestdoi/481248275)
# qrpca(R package)

qrpca behaves similarly prcomp. But employs a QR-based PCA instead of applying singular value decomposition on the original matrix. The code uses torch under the hood for matrix operations and supports GPU acceleration.

## Installation

Source installation from GitHub:

```R
install.packages('remotes')
remotes::install_github("RafaelSdeSouza/qrpca")
library(qrpca)
```
## Usage

An example of using the package to run a PCA:

``` r
library(qrpca)
set.seed(42)
  N <- 1e4
  M <- 1e3
  X <- matrix(rnorm(N*M), M)
  system.time(prcomp(X))
  system.time(qrpca(X))
  system.time(qrpca(X,cuda = TRUE))
```
 For an astronomical example, we use a datacube from MaNGA for galaxy. It comprises a tensor [74,74,4563], of two spatial dimensions and one spectral dimension. The following code reads the cube and flattens the spectra into a matrix of dimension [5476,4563]
 
 ``` r
require(qrpca);require(reticulate)
require(FITSio);require(ggplot2);
require(dplyr);require(reshape2)
cube <- "manga-7443-12703-LOGCUBE.fits"
df <- readFITS(cube)
n_row <- dim(df$imDat)[1]
n_col <- dim(df$imDat)[2]
n_wave <- dim(df$imDat)[3]
data.2D  <- array_reshape(df$imDat,
c(n_row*n_col,n_wave),order = c("F"))
pca  <- qrpca(data.2D)
# Function to extract the k-th eigenmap
eigenmap <- function(pcobj, k = 1){
  x <- as.matrix(pcobj$x)
  out <- matrix(x[,k],nrow=n_row,ncol=n_col)
  out
}

map <- eigenmap(pca) %>% melt()

ggplot(map,aes(x=Var1,y=Var2,z=value)) +
  geom_raster(aes(fill=value)) +
  scale_fill_viridis_c(option="C") +
  theme(legend.position = "none") 
 ```





## Dependencies

`torch`

## References
- Sharma, Alok and Paliwal, Kuldip K. and Imoto, Seiya and Miyano, Satoru 2013, International Journal of Machine Learning and Cybernetics, 4, 6, doi: [10.1007/s13042-012-0131-7](https://doi.org/10.1007/s13042-012-0131-7)
