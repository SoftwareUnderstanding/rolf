# spops

Data release supporting:

* *Spin orientations of merging black holes formed from the evolution of stellar binaries.*
  Davide Gerosa, Emanuele Berti, Richard O’Shaughnessy, Krzysztof Belczynski, Michael Kesden, Daniel Wysocki, Wojciech Gladysz. Physical Review D 98 (2018) [084036](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.084036). arXiv:[1808.02491](http://arxiv.org/abs/arXiv:1808.02491) [astro-ph.HE].
* *Multiband gravitational-wave event rates and stellar physics.*
  Davide Gerosa, Sizheng Ma, Kaze W.K. Wong, Emanuele Berti, Richard O’Shaughnessy, Yanbei Chen, Krzysztof Belczynski. arXiv:[1902.00021](http://arxiv.org/abs/arXiv:1902.00021) [astro-ph.HE].

Here provide our database and a short python code to query it.

### Credit

You are more than welcome to use our database in your research; we kindly ask you to cite our papers above. For questions or bugs, just ask [me](www.davidegerosa.com). Oh, and before you ask, **SPopS** means "**S**pinning black-hole binary **POP**ulation **S**ynthesis". If you want to cite the database specifically, it's
[![DOI](https://zenodo.org/badge/142477838.svg)](https://zenodo.org/badge/latestdoi/142477838)


### Examples

To access the effective spin distribution and the LIGO detection rates of one specific model with python:

    import spops
    db=spops.database()
    model = {"kicks":"70", "spins":"collapse", "tides":"time", "detector":"LIGO"}
    var='chieff'
    print(db(model,var))  
    var='detectionrate'
    print(db(model,var))

For the same population synthesis simulation, this is the mass ratio of black hole binaries detectable by a multiband LISA+CosmicExplorer network

    model = {"kicks":"70", "spins":"collapse", "tides":"time", "detector":"LISACosmicExplorer", "Tobs":"10", "SNRthr":"8"}
    var='q'
    print(db(model,var))
    var='detectionrate'
    print(db(model,var))



### Database

We provide a database in `h5` format containing all population sysnthesis distributions perfomed with [StarTrack](https://www.syntheticuniverse.org/) and post-processed with [precession](https://davidegerosa.com/precession/).

The database's size is ~17GB, and needs to be downloaded in chunks from the [GitHub release page](https://github.com/dgerosa/spops/releases). Execute the following:

    for i in $(seq -f "%02g" 0 16); do
    wget "https://github.com/dgerosa/spops/releases/download/v0.2/spops.h5_"$i;
    done
    cat spops.h5_* > spops.h5; rm spops.h5_*

(on mac: `brew install wget').

Models are described by the following options:
  - **kicks**. Magnitude of the Supernova kicks. Available options are `['0','25','50','70','130','200','265']`.
  - **spins**. Prescription for the spin magnitudes. Available options are `['collapse','max','uniform']`.
  - **tides**. Prescription for tidal spin alignment. Available options are `['time','alltides','notides']`.
  - **detector**. Targeted detector or multiband network. Available options are `['LIGO','Voyager','CosmicExplorer','LISA','LISALIGO','LISACosmicExplorer']`.
  - **Tobs**. Duration of the LISA mission in yrs.  Available options are `['4','10]`.
  - **SNRthr**. LISA SNR threshold.  Available options are `['4','8]`.

Not all options are required for all the variables (for instance, LISA duration does not need to be specified to access ground-only information).


The following variables are available:

  - `Mzams_a`: ZAMS mass of the heavier star (solar masses).
  - `Mzams_b`: ZAMS mass of the lighter star (solar masses).
  - `M_a`: Mass of the black hole formed by the heavier ZAMS star (solar masses).
  - `M_b`: Mass of the black hole formed by the lighter ZAMS star (solar masses).
  - `zmer`: Merger redshift.
  - `met`: Metallicity.
  - `path`: Formation pathway, convention in Sec. 2D of arXiv:[1808.02491](http://arxiv.org/abs/arXiv:1808.02491).
  - `tidealign`: boolean flag marking if the stellar spin was realigned by tides.
  - `M`: Black-hole binary total mass (solar masses).
  - `q`: Black-hole binary mass ratio.
  - `chi1`: Spin of the heavier black hole.
  - `chi2`: Spin of the lighter black hole.
  - `chieff`: Black-hole binary effective spin
  - `morph`: Spin morphology (-1: Librating about 0, 0: Circulating, +1: Librating about pi)
  - `theta1`: Tilt angle of the heavier black hole at 20 Hz.
  - `theta2`: Tilt angle of the heavier black hole at 20 Hz.
  - `deltaphi`: Difference between the azimuthal spin angles at 20 Hz.
  - `SNR`: Signal-to-noise ratio. For ground-based detector returns the optimal SNR. For LISA returns SNR evaluated at Tobs.
  - `detectionrate`: Detection rate for the targeted detector.


### Python module

We also provide a simple python module to query the database. `spops` is compatible with both Python 2 and Python 3 and can be installed from the [Python Package index](https://pypi.python.org/pypi/spops) using:

    pip install spops

Remember to download and assemble the database as described above. The module contains a single class, called `database`. To initialize the class:

    import spops
    db = spops.database(h5filename='spops.h5',h5dir=None)

The input parameters are:

  - `h5filename`: database file name, default is `spops.h5`.
  - `h5dir`: directory of the database; if `None` (default) the code will look for detabase in both the location where the `spops` module is installed (this is `os.path.dirname(os.path.abspath(spops.__file__))`) and the execution location (this is `.`).

The population sysnthesis run of interest can be specified using a python dictionary with the keys as above, so for instance

    model = {"kicks":"70", "spins":"collapse", "tides":"time", "detector":"LIGO"}

One can then access the datasets described above by just calling the database class:

    var='chieff'
    print(db(model,var))

To list the model options and the available variables use

    print(db.options)
    print(db.vars)

#### A few technical notes:

  - The `database` class is a [singleton](https://en.wikipedia.org/wiki/Singleton_pattern): only one istance can exist at any time. Multiple calls will return pointers to the same instance. This is done to prevent useless memory allocation. For instance:

        db1=spops.database()
        db2=spops.database()
        print(db1==db2)


        >>>> True

  - We do [lazy loading](https://en.wikipedia.org/wiki/Lazy_loading) here and only read in a dataset when/if the user asks for it.  Moreover, `spops` remembers which dataset have already been loaded, such that each subsequent access is read in from memory, not disk. So for instance:

        from contexttimer import timer
        @timer()
        def read_from_spops(model,var):
            return db(model,var)
        var='Mzams_a'
        read_from_spops(model,var)
        read_from_spops(model,var)

        >>>> function read_from_spops execution time: 0.002
        >>>> function read_from_spops execution time: 0.000
