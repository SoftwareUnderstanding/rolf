# NIMBLE (**N**on-parametr**I**c jeans **M**odeling with **B**-sp**L**in**E**s)

NIMBLE is a tool for inferring the cumulative mass distribution of a gravitating system from full 6D phase space coordinates of its tracers via spherical Jeans modeling. Spherical Jeans modeling inherently assumes the system is spherically symmetric and in dynamical equilibrium, however, Rehemtulla et al. 2022 show that these conditions are not completely necessary for an accurate mass profile estimate when applied mock Milky Way-like galaxies.

Rehemtulla et al. 2022 (https://arxiv.org/abs/2202.05440) gives much more detail on the routines included here and extensive tests using them.

NIMBLE also includes codes for performing related tasks:

- Creating a variety of equilibrium mock galaxies using the Agama library (https://github.com/GalacticDynamics-Oxford/Agama)
- Creating mock Gaia & DESI observational data of the Latte suite of FIRE-2 cosmological hydrodynamic zoom-in simulations (Wetzel+2016, https://ui.adsabs.harvard.edu/abs/2016ApJ...827L..23W)

### Testing with `halo alone` datasets

NIMBLE includes a bash script titled `run_halo_alone.sh` which will generate, run Jeans modeling on, and plot a figure for `halo alone` type datasets of up to 3 provided axis ratios.

The `halo alone` datasets are generated using Agama (https://github.com/GalacticDynamics-Oxford/Agama) through `halo_alone.py` located in `equilibrium_models/`. For example, to create a `halo alone` dataset with q=0.8 run the following:

```bash
python3 equilibrium_models/halo_alone.py 0.8
```

This will create `halo_alone_0.8_prejeans.csv` and `halo_alone_0.8_true.csv` in `data/halo_alone`, which contain kinematic information of an N-body representation of this model. These files are used in the following step where the NIMBLE inverse modeling Jeans routine is executed on the dataset using `jeans_bspline.py`. To run it, provide file paths to the `_prejeans.csv`  and `_true.csv` files created in the previous step.

```bash
python3 jeans_bspline.py data/halo_alone/halo_alone_0.8_prejeans.csv data/halo_alone/halo_alone_0.8_true.csv
```

This will create an assortment of files in `results/halo_alone_0.8/` including plots of the velocity, density, velocity anisotropy, and mass enclosed profiles. To create a figure comparing the results of multiple `halo alone` runs, similar to Fig. 3 in Rehemtulla+2022, run `fig3-5.py` located in `figures/` with the argument `halo_alone`.

### Testing with `halo_disk_bulge` datasets

Running `halo_disk_bulge` is very similar to running `halo_alone`. The `run_halo_disk_bulge.sh` script will generate, run Jeans modeling on, and plot a figure for the original dataset and its two variants (described in Sec. 3.1 of Rehemtulla+2022).

To generate these datasets manually, use `halo_disk_bulge.py` in `equilibrium_models/` as follows:

```bash
python3 equilibrium_models/halo_disk_bulge.py OM
```

In this example I've optionally added `OM` which creates the variant with a Cuddeford-Osipkov-Meritt velocity anisotropy profile. Omit `OM` and you'll generate the original HDB dataset alongside its disk contamination variant, again described in Sec. 3.1 of Rehemtulla+2022.

The process of running NIMBLE's inverse modeling Jeans routine on these and plotting a comparison figure is done identically for these and the `halo_alone` datasets. Do note that the disk contamination variants share a `_true.csv` file with their non-disk contamination parents so there will be no `HDB_DC_true.csv` file.

### Testing with Latte FIRE-2 galaxies (without observational effects)

Running the inverse modeling routine on error-free Latte data comes with the additional complication of having to download the Latte data. `run_latte_errorfree.sh` automates downloading it from [yt Hub](https://girder.hub.yt/#collection/5b0427b2e9914800018237da/folder/5b211e42323d120001c7a813) but it is quite large so it will still take some time. Once the data is downloaded for each Latte galaxy of interest, `run_latte_errorfree.sh` will prepare the mocks, run the inverse modeling Jeans routine on them, and plot a figure of the resulting mass profiles.

As usual, all these steps can be performed manually with the individual scripts if desired.

```bash
python3 read_latte.py m12f
```

Running the above command will create the `m12f_prejeans.csv` and `m12f_true.csv` files used for Jeans modeling. This will also require the `gizmo_read` package, also available at [yt Hub](https://girder.hub.yt/#collection/5b0427b2e9914800018237da/folder/5b211e42323d120001c7a813) and also automatically set up by `run_latte_errorfree.sh`. You can then run the inverse modeling Jeans routine in `jeans_bspline.py` similarly to `halo_alone` datasets by providing the paths to these two `.csv` files. The Jeans routine will automatically select the knot configurations shown in Rehemtulla+2022 but can still be configured to use custom knots. Creating a figure comparing the results from the three Latte galaxies can be done by running `figures/fig3-5.py` with the argument `latte`.

### Testing with Latte FIRE-2 galaxies (with observational effects and deconvolution)

Running the deconvolution routine with observational selection functions and errors imposed on Latte data is a little more involved than running the inverse modeling routine. `deconv.py` will do most of the work, however. It handles imposing the observational effects, deconvolving them, and outputting the Jeans estimated mass proifle. It's output can then be used in `figures/fig7.py` to make a copy of Fig. 7 in Rehemtulla+2022. Before running `deconv.py`, one must download and prepare the Latte data (as described in the previous section) and install a few packages.

The output of each MCMC trial and the final deconvolution output will be saved in a directory for each run. Taking m12f LSR1 with DR4 errors as an example, output will be saved to `results/deconv_m12f_LSR1_DR4/`. Before the first MCMC trial and after each following one, `deconv.py` will create figures showing the true, error-imposed, and MCMC-deconvolved density and velocity dispersion profiles. It will also make a corner plot and an MCMC parameter evolution plot after each trial. After convergence, a few text files are written. Most importantly, `Menc_beta_final.csv` will contain a radial grid and the 16-50-84 percentiles for the enclosed mass and velocity anisotropy profiles.

There is a bash shell script titled `run_deconv.sh` included to automate the deconvolution routine. When using it, make sure that the configurations `deconv.py` is run on match those chosen to be plotted in `fig7.py` (i.e. `sims` in `run_deconv.sh` matches `sims_to_plot` in `fig7.py` and similarly for `lsrs`)

To run `deconv.py` and `fig7.py` manually use the following commands.
Again taking m12f LSR1 with DR4 errors as an example

```bash
python3 deconv.py m12f lsr1 dr4
python3 fig7.py dr4
```

#### Requirements
Matplotlib 3.0.0 or newer
Python 3.6 or newer
