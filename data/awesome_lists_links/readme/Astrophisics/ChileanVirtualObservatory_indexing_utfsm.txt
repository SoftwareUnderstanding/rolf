Indexación de objetos astronómicos
==================================

R codes.

- cube_indexing.R runs the full pipeline of the system. 

It has dependences on EBImage, FITSio, ptw, and R. 

Source codes are organized as follows:

- Spectra processing codes are in the following files: accumulating.R, cube_spectra.R, differenting.R, erosing.R, masking.R, pixel_processing.R, reading_cubes.R, segmenting.R, stacking.R, and vel_stacking.R

- Multiscale segmentation codes are in the following files: gaussian_mix.R, bg_fg.R, optimal_w.R, kernelsmooth.R, kernel_shift.R

- Indexing codes (to deal with a relational database): mainFunction.R (runs the indexing pipeline), getObjects.R, getCoordData.R, calculateCoords.R, and writeToDataBase.R


Examples

- ROI detection:

R> spectra <- cube_spectra(Z_block,500)

R> h1 <- vel_stacking(Z_block,67,96)

R> h2 <- vel_stacking(Z_block,105,148)

- Multiscale segmentation:

R> gaussian_mix(h1)

R> gaussian_mix(h2)

R> getObjects(h1)

R> getObjects(h2)

- Relational indexing:

% Dependences on DBI and RPostgreSQL 

R> drv <- dbDriver("PostgreSQL")

R> database <- dbConnect(drv, dbname="alma", user="postgres", password=" ")

R> mainFunction(database,fits)

@Marcelo Mendoza (21/4/15)
