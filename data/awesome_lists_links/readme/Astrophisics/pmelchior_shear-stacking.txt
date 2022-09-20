# shear-stacking-tests

This repository contains python scripts to calculate stacked shear profiles and tests based upon them, e.g. consistency for different slices of lensed background galaxies. The basic concept is that the lensing signal in terms of surface mass density (instead of shear) should be entirely determined by the properties of the lens sample and have no dependence on source galaxy properties. An example application is given in section 4.4 of [Melchior et al. (2015)](http://arxiv.org/abs/1405.4285).

## Usage

``` 
python run_quadrant_check.py test/config.json
python create_profiles.py test/config.json shear
python plot_profiles.py test/config.json shear
```

This is a typical session. The first command performs the quadrant check, which ensures that each lens in the respective catalog has at least 2 adjacent quadrants of sources within the `maxrange` of the stacks to ensure cancellation of additive shear systematics. This step is optional but recommended. It create another fits file, and you will need to add it as `lens_extra_file` to the config file.

The second command does the heavy lifting. It opens shape and lens catalogs (including any extra files for either), splits the shape catalog into several chunks of `shape_chunk_size` elements, and distributes the chunks over the available CPUs with pythons `multiprocessing`. Each chunk of sources is then matched against all lenses, and the pairs are put into radial profiles. If additional `splitting` are defined, a profile will be created for each slice.

If `n_jack > 0`, then the script will internally run spatial jackknife for the lenses. It will make use a jackknife region file, which is placed in `n_jack/km_centers.npy`, which maps each lens index to a spatial region. The profiles for each region (i.e. those whose lens are _not_ in that region), will also be stored in the `n_jack` subdirectory.

At the end, the script creates a number of profile files in pythons `npz` format, which contain the following keys/files:

* `mean_r`: density-weighted center of radial bins, units are [Mpc / h] if `coords=='physical'` and [arcmin] if `coords=='angular'`
* `n`: number of pairs
* `mean_q`: Mean of the stacked quantity, see below. Units are [10^14 M_solar / Mpc^2] or the natural units of `shape_scalar_key`.
* `std_q`: In-bin standard deviation or Jackknifed error of `mean_q`
* `sum_w`: the total weight of all pairs, in units of Sigma_crit^-2 or of `shape_weight_key`

The script can run with two different types of profile: `shear` or `scalar`. In the former case, it will compute the tangential shear from the columns `shape_e1_key`, `shape_e2_key`, `shape_z_key`, `shape_weight_key`, `shape_sensitivity_key`. In the latter case, it create a radial profile of the scalar quantity denotes by `shape_scalar_key` and its weight `shape_weight_key` (no sensitivity is considered).

The third command simply takes the `.npz` files and creates the desired plots.

## Configuration file format

Virtually all aspects of the script can be controlled from a config file, in json format. This way, the scripts do not have to be altered to adjust to the pecularities of the lens or shape catalogs.

An example for a shear profile is given below:

``` json
{
        "coords": "angular",
        "maxrange": 1.1,
        "n_jack": 40,
        "lens_catalog": "/catalogs/redmapper/redmapper_catalog.fit",
        "lens_cuts": [],
        "lens_z_key": "Z_LAMBDA",
        "shape_file": "/catalogs/im3shapev7_ngmix009/shapes.fits.gz",
        "shape_z_key": "ZP",
        "shape_ra_key": "ALPHAWIN_J2000_R",
        "shape_dec_key": "DELTAWIN_J2000_R",
        "shape_e1_key": "im3shape_r_e1",
        "shape_e2_key": "im3shape_r_e2",
        "shape_weight_key": "weight_im3shape",
        "shape_sensitivity_key": "nbc_m",
        "shape_cuts": [
                "MODEST_CLASS == 1",
                "im3shape_r_exists == 1",
                "im3shape_r_error_flag == 0"
        ],
        "split_type": "shape",
        "splittings": {
                "FLAGS_I": [0, 1, 2, 4],
                "ZP": [0.7, 0.9, 1.1, 1.5],
                "B_D": [0.0, 0.3, 0.7, 1.0]
        },
        "functions": {
                "weight_im3shape": "0.2**2/(0.2**2 + (0.1*20/s['im3shape_r_snr'])**2)",
                "nbc_m": "1 + s['im3shape_r_nbc_m']",
                "B_D": "s['im3shape_r_bulge_flux'] / s['im3shape_r_disc_flux']"
        }
}
```

Coordinates can be either `angular` or `physical` and refer to the units of the radial profiles. `maxrange` is the maximum extent of the profile, in deg or Mpc/h (depending on `coords`).

`lens_cuts` and `shape_cuts` affect which objects are loaded from either lensing or shape catalogs; filtering can be used on all columns available in either fits files.

`splitting` denote the kind of slices of the either shape or lens catalogs (determined by the value of `split_type`). For technical reason, there need to be at least two splitting keys, or none. The keys of this dictionary specify either a column in the respective catalog or an entry in the `functions` list. For the latter, any function based on the catalog (denoted as `s`) can be implemented. The values denote the limits of the slices for each key, with the upper limit being excluded, e.g. `"FLAGS_I": [0, 1, 2, 4]` creates three slices:

``` 
FLAGS_I in [0,1), [1,2), [2,4) 
```

There is no formal limit on how many different categories/keys can be done, but the number of slices should not exceed 5 (otherwise the plots get rather busy).

User-defined `functions` can be used for any `_key` entry in the config file, both for lens and for shape catalogs.

## Dependencies

* [multiprocessing](https://docs.python.org/2/library/multiprocessing.html)
* [Erin Sheldon's esutil](https://code.google.com/p/esutil/)
* [Erin Sheldon's fitsio](https://github.com/esheldon/fitsio)
* matplotlib