# mc-spam

MC-SPAM (Monte-Carlo Synthetic-Photometry/Atmosphere-Model) is an algorithm to generate limb-darkening coefficients 
from models that are comparable to transit photometry according to the formalism described in Espinoza & Jordan (2015), 
which improves the original SPAM algorithm proposed by Howarth (2011) by taking in consideration the uncertainty on 
the stellar and transit parameters of the system under analysis.

If you use this code for your research, please consider citing Espinoza & Jordan (2015; http://arxiv.org/abs/1503.07020).

DEPENDENCIES
------------

This code makes use of three important libraries:

        + Numpy.
        + Scipy.
        + Pyfits.

All of them are open source and can be easily installed in any machine.

INSTALLATION
------------

Because this code uses transit modelling to obtain the MC-SPAM limb-darkening coefficients, it needs an implementation 
of the Mandel & Agol (2002) transit modelling in order to run. For this, a Fortran implementation of the code is under 
the "main_codes" folder, which in turn is called by our Python routines. In order for this link to be made, you 
need to "install" the package by simply running:

		python install.py

Which will automatically generate the needed files for the transit code to run.

USAGE
-----

In order to run, the code needs to know the following parameters of a given system:

        p:              The planet-to-star radius ratio, Rp/R_*.

        i:              The inclination of the orbit (in degrees).

        aR:             The semi-major axis to stellar radius ratio (a/R_*)

        e:              The eccentricity of the orbit.

        omega:          The argument of periastron (in degrees).

        Teff:           Effective temperature of the host star (in Kelvins)

        logg:           Log-gravity of the host star.

        MH:             Metallicity of the host star (~[Fe/H]).

        vturb:          Microturbulent velocity of the host star (in km/s).


These parameters can be either estimated, in which case you need the associated uncertainties, 
fixed or obtained through an MCMC chain. If you have any data for your system that was estimated 
by previous works (or from you and for which you do not have an MCMC chain), you must input it 
under the "estimated_parameters" folder; the "planet_data.dat" stores the parameters of the transit, 
while "star_data.dat" stores the parameters of the host star (note that the names of both systems 
must match; see the files for example inputs). If a parameter is fixed by some reason, fix their 
upper and lower errors to zero. 

If you want to use an MCMC chain for a given parameter, input any value for that parameter 
in the above mentioned files and modify the "get_mcspam_vals.py" file in order to input 
your MCMC chains (see the example under lines 76 to 83 of the "get_mcspam_vals.py" code).

After all of the above is set, you can edit the options in the top part of the 
"get_mcspam_vals.py" file and run it by simply doing:

		python get_mcspam_vals.py

This will then generate the MC-SPAM estimates of the model limb-darkening coefficients.

OUTPUTS
-------

The program will generate an output folder with a user-defined name (the default is "results"), 
in which a folder for each system will be created along with a mc_spam_results.dat file that 
will contain the 0.16, 0.5 and 0.84 quantiles (i.e., the median and the "1-sigma" errors) of 
the distribution of both the model and the MC-SPAM estimates of the limb-darkening coefficients. 
Inside each folder, the Monte-Carlo samples of both the original model and the estimated MC-SPAM 
limb-darkening coefficients will be saved as FITS files. 

