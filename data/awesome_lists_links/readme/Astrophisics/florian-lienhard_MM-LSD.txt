# MM-LSD

Multi-Mask Least-Squares Deconvolution

#### email:  fl386_at_cam.ac.uk
#### First Version 28 April 2022


## Paper
### F. Lienhard, A. Mortier, L. Buchhave, A. Collier Cameron, M. López-Morales, A. Sozzetti, C. A. Watson, R. Cosentino

### [MNRAS](https://doi.org/10.1093/mnras/stac1098), [arXiv](https://arxiv.org/abs/2204.13556)



#### Brief description of MM-LSD:

* The code runs in _python 3_.
* Continuum normalise 2D spectra (echelle order spectra).
* Mask and partially correct telluric lines.
* Extract RVs from spectra using Least-Squares Deconvolution using MM-LSD technique described in Lienhard et al. 2022.

#### Check MM-LSD [wiki](https://github.com/florian-lienhard/MM-LSD/wiki)



####
* RASSINE:
[Cretignier et al. 2020](https://www.aanda.org/articles/aa/pdf/2020/08/aa37722-20.pdf),
[Github](https://github.com/MichaelCretignier/Rassine_public)
* VALD3:
[Ryabchikova et al. 2015](https://iopscience.iop.org/article/10.1088/0031-8949/90/5/054005),
[Website](http://vald.astro.uu.se/)
* TAPAS:
[Bertaux et al. 2014](https://zenodo.org/record/11110#.YmgXlFzMIfA),
[Website](http://cds-espri.ipsl.fr/tapas/)

The definition of RVerror in classes.py is based on https://github.com/j-faria/iCCF/.
