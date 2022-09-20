# Fourier dimensionality reduction for radio interferometry

Code repository for dimensionality reduction techniques proposed for radio-interferometric data.

MATLAB programs for tests on real and simulated data are made available here. For more details of the mathematical background and corresponding results, please refer to the following articles:

## Real data:  Robust dimensionality reduction for interferometric imaging of Cygnus A

Please run the tests from `realdata_*_pdfb.m`.

Results are automatically stored in corresponding subdirectories.

#### Based on:
["Robust dimensionality reduction for interferometric imaging of Cygnus A"](https://arxiv.org/abs/1709.03950)

Submitted to Monthly Notices of the Royal Astronomical Society

## Simulated data: A Fourier dimensionality reduction model for big data interferometric imaging

Please run `example_reconstruction_admm.m` to reconstruct M31 images with reduced data.

Results and logging details are found in corresponding subdirectories.

#### Based on:
["A Fourier dimensionality reduction model for big data interferometric imaging"](http://arxiv.org/abs/1609.02097)

Monthly Notices of the Royal Astronomical Society (2017) [468 (2): 2382-2400](https://doi.org/10.1093/mnras/stx531).

## Contents

1. MATLAB code implementing Fourier-based dimensionality reduction of interferometric data
2. Tests running on real interferometer data, with different dimensionality reduction methods
3. Example tests running M31 image reconstruction
4. Coverage data
5. Test images M31 and Cygnus A

#### Dependency: Nonuniform FFT toolbox [(Download toolbox)](http://web.eecs.umich.edu/~fessler/irt/fessler.tgz) [(Read reference article)](http://dx.doi.org/10.1109/TSP.2002.807005)


### Authors and Contributors

S. Vijay Kartik ([@vijaykartik](https://github.com/vijaykartik)), Rafael E. Carrillo, Arwa Dabbech, Jean-Philippe Thiran and Yves Wiaux

### Support or Contact

S. Vijay Kartik ([@vijaykartik](https://github.com/vijaykartik))
