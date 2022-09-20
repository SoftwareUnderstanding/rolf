# BANYAN Σ (IDL)
A Bayesian classifier to identify members of the 27 nearest young associations within 150 pc of the Sun.

This is the IDL version of BANYAN Σ. The Python version can be found at https://github.com/jgagneastro/banyan_sigma

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1165086.svg)](https://doi.org/10.5281/zenodo.1165086)

## PURPOSE:
Calculate the membership probability that a given astrophysical object belongs to one of the currently known 27 young associations within 150 pc of the Sun, using Bayesian inference. This tool uses the sky position and proper motion measurements of an object, with optional radial velocity (RV) and distance (D) measurements, to derive a Bayesian membership probability. By default, the priors are adjusted such that a probability treshold of 90% will recover 50%, 68%, 82% or 90% of true association members depending on what observables are input (only sky position and proper motion, with RV, with D, with both RV and D, respectively).
       
Please see Gagné et al. 2018 (accepted for publication in ApJS, http://adsabs.harvard.edu/abs/2018arXiv180109051G) for more detail.
       
An online version of this tool is available for 1-object queries at http://www.exoplanetes.umontreal.ca/banyan/banyansigma.php.
       
## REQUIREMENTS:
(1) A fits file containing the parameters of the multivariate Gaussian models of each Bayesian hypothesis must be included at /data/banyan_sigma_parameters.fits in the directory where BANYAN_SIGMA.pro is compiled. The file provided with this release corresponds to the set of 27 young associations described in Gagné et al. (2018). The fits file can be written with MWRFITS and must contain an IDL array of structures of N elements, where N is the total number of multivariate Gaussians used in the models of all Bayesian hypotheses. Each element of this structure contains the following information:

        NAME: The name of the model (scalar string).
        CENTER_VEC: Central XYZUVW position of the model (6D vector, in units of pc and km/s).
        COVARIANCE_MATRIX: Covariance matrix in XYZUVW associated with the model (6x6 matrix, in mixed units of pc and km/s).
        PRECISION_MATRIX: (Optional) Matrix inverse of COVARIANCE_MATRIX, to avoid re-calculating it many times (6x6 matrix).
        LN_NOBJ: (Optional) Natural logarithm of the number of objects used to build the synthetic model (scalar). This is not used in BANYAN_SIGMA.
        COVARIANCE_DETERM: (Optional) Determinant of the covariance matrix, to avoid re-calculating it many times (scalar).
        PRECISION_DETERM: (Optional) Determinant of the precision matrix, to avoid re-calculating it many times (scalar).   
        LN_ALPHA_K: (Optional) Natural logarithm of the alpha_k inflation factors that ensured a fixed rate of true positives at a given Bayesian probability treshold. See Gagné et al. (2018) for more detail (scalar or 4-elements vector). This is not used in BANYAN_SIGMA.
        LN_PRIOR: Natural logarithm of the Bayesian prior (scalar of 4-elements vector). When this is a 4-elements vector, the cases with only proper motion, proper motion + radial velocity, proper motion + distance or proper motion + radial velocity + distance will be used with the corresponding element of the LN_PRIOR vector.
        LN_PRIOR_OBSERVABLES: Scalar string or 4-elements vector describing the observing modes used for each element of LN_PRIOR. This is not used in BANYAN_SIGMA.
        COEFFICIENT: Coefficient (or weight) for multivariate Gaussian mixture models. This will only be used if more than one element of the parameters array have the same model name (see below). When more than one elements have the same model name, BANYAN_SIGMA will use the COEFFICIENTs to merge its Bayesian probability, therefore representing the hypothesis with a multivariate Gaussian model mixture. This is how the Galactic field is represented in Gagné et al. (2018).

(2) (Optional) A fits file containing the various performance metrics (true positive rate, false positive rate, positive predictive value) as a function of the Bayesian probability treshold, for each young association. Each element of this structure contains the following information:

        NAME: The name of the model (scalar string).
        PROBS: N-elements array containing a list of Bayesian probabilities (%).
        TPR: Nx4-elements array containing the rate of true positives that correspond to each of the Bayesian probability (lower) tresholds stored in PROBS.
        FPR: Nx4-elements array containing the rate  of false positives that correspond to each of the Bayesian probability (lower) tresholds stored in PROBS.
        PPV: Nx4-elements array containing the Positive Predictive Values that correspond to each of the Bayesian probability (lower) tresholds stored in PROBS.
        NFP: Number of expected false positives (FPR times the ~7 million stars in the Besancon simulation of the Solar neighborhood) 
           
Each component of the 4-elements dimension of TPR, FPR, NFP and PPV corresponds to a different mode of input data, see the description of "LN_PRIOR" above for more detail.
           
When this fits file is used, the Bayesian probabilities of each star will be associated with a TPR, FPR, NFP and PPV values in the METRICS sub-structure of the output structure.
           
This file must be located at /data/banyan_sigma_metrics.fits in the directory where BANYAN_SIGMA.pro is compiled. The file provided with this release corresponds to the set of models described in Gagné et al. (2018).
           
## CALLING SEQUENCE:
 
```idl
OUTPUT_STRUCTURE = BANYAN_SIGMA([ stars_data, COLUMN_NAMES=column_names, HYPOTHESES=HYPOTHESES, LN_PRIORS=LN_PRIORS, NTARGETS_MAX=ntargets_max, RA=RA, DEC=DEC, PMRA=PMRA, PMDEC=PMDEC, EPMRA=EPMRA, EPMDEC=EPMDEC, DIST=DIST, EDIST=EDIST, RV=RV, ERV=ERV, PSIRA=PSIRA, PSIDEC=PSIDEC, EPSIRA=EPSIRA, EPSIDEC=EPSIDEC, PLX=PLX, EPLX=EPLX, CONSTRAINT_DIST_PER_HYP=CONSTRAINT_DIST_PER_HYP, CONSTRAINT_EDIST_PER_HYP=CONSTRAINT_EDIST_PER_HYP, /UNIT_PRIORS, /LNP_ONLY, /NO_XYZ, /USE_RV, /USE_DIST, /USE_PLX, /USE_PSI ])
```
 
## OPTIONAL INPUTS:
        stars_data: An IDL structure (or array of structures when more than one objects are analyzed) that contain at least the following tags: RA, DEC, PMRA, PMDEC, EPMRA, and EPMDEC. It can also optionally contain the tags RV, ERV, DIST, EDIST, PLX, EPLX, PSIRA, PSIDEC, EPSIRA, EPSIDEC. See the corresponding keyword descriptions for more information. If this input is not used, the keywords RA, DEC, PMRA, PMDEC, EPMRA, and EPMDEC must all be specified.
        column_names: An IDL structure that contains the names of the "stars_data" columns columns which differ from the default values listed above. For example, column_names = {RA:'ICRS_RA'} can be used to specify that the RA values are listed in the column of stars_data named ICRS_RA.
        RA: Right ascension (decimal degrees). A N-elements array can be specified to calculate the Bayesian probability of several stars at once, but then all mandatory inputs must also be N-elements arrays.
        DEC: Declination (decimal degrees).
        PMRA: Proper motion in the right ascension direction (mas/yr, must include the cos(dec) factor).
        PMDEC: Proper motion in the declination direction (mas/yr).
        EPMRA: Measurement error on the proper motion in the right ascension direction (mas/yr, must not include the cos(dec) factor).
        EPMDEC:  Measurement error on the proper motion in the declination direction (mas/yr).
        RV: Radial velocity measurement to be included in the Bayesian probability (km/s). If this keyword is set, ERV must also be set. A N-elements array must be used if N stars are analyzed at once.
        ERV: Measurement error on the radial velocity to be included in the Bayesian probability (km/s). A N-elements array must be used if N stars are analyzed at once.
        DIST: Distance measurement to be included in the Bayesian probability (pc). By default, the BANYAN_SIGMA Bayesian priors are meant for this keyword to be used with trigonometric distances only. Otherwise, the rate of true positives may be far from the nominal values described in Gagné et al. (2018). If this keyword is set, EDIST must also be set. A N-elements array must be used if N stars are analyzed at once.
        EDIST: Measurement error on the distance to be included in the Bayesian probability (pc). A N-elements array must be used if N stars are analyzed at once.
        PLX: Parallax measurement to be included in the Bayesian probability (mas). The distance will be approximated with DIST = 1000/PLX. If this keyword is set, EPLX must also be set. A N-elements array must be used if N stars are analyzed at once.
        EPLX: Measurement error on the parallax to be included in the Bayesian probability (mas). The distance error will be approximated with EDIST = 1000/PLX^2*EPLX. A N-elements array must be used if N stars are analyzed at once.
        PSIRA: Parallax motion factor PSIRA described in Gagné et al. (2018), in units of 1/yr. If this keyword is set, the corresponding PSIDEC, EPSIRA and EPSIDEC keywords must also be set. This measurement is only useful when proper motions are estimated from two single-epoch astrometric measurements. It captures the dependence of parallax motion as a function of distance, and allows BANYAN_SIGMA to shift the UVW center of the moving group models, which is equivalent to correctly treating the input "proper motion" PMRA, PMDEC, EPMRA, EPMDEC as a true apparent motion. This keyword should *not* be used if proper motions were derived from more than two epochs, or if they were obtained from a full parallax solution. A N-elements array must be used if N stars are analyzed at once.
        PSIDEC: Parallax motion factor PSIDEC described in Gagné et al. (2018), in units of 1/yr. A N-elements array must be used if N stars are analyzed at once.
        EPSIRA: Measurement error on the parallax motion factor PSIRA described in Gagné et al. (2018), in units of 1/yr. A N-elements array must be used if N stars are analyzed at once.
        EPSIDEC: Measurement error on the parallax motion factor PSIDEC described in Gagné et al. (2018), in units of 1/yr. A N-elements array must be used if N stars are analyzed at once.
        NTARGETS_MAX: (default 10^6). Maximum number of objects to run at once in BANYAN_SIGMA to avoid saturating the RAM. If more targets are supplied, BANYAN_SIGMA runs over a loop of several batches of NTARGETS_MAX objects. 
        HYPOTHESES: The list of Bayesian hypotheses to be considered. They must all be present in the parameters fits file (See REQUIREMENTS #1 above).
        LN_PRIORS: An IDL structure that contains the natural logarithm of Bayesian priors that should be *multiplied with the default priors* (use /UNIT_PRIORS if you want only LN_PRIORS to be considered). The structure must contain the name of each hypothesis as tags, and the associated scalar value of the natural logarithm of the Bayesian prior for each tag. 
        CONSTRAINT_DIST_PER_HYP: An IDL structure (or array of IDL structures when several objects are analyzed) that contains a distance constraint (in pc). Each of the Bayesian hypotheses must be included as structure tags and the distance must be specified as its associated scalar value. CONSTRAINT_EDIST_PER_HYP must also be specified if CONSTRAINT_DIST_PER_HYP is specified. This keyword is useful for including spectro-photometric distance constraints that depend on the age of the young association or field.
        CONSTRAINT_EDIST_PER_HYP: An IDL structure (or array of IDL structures when several objects are analyzed) that contains a measurement error on the distance constraint (in pc). Each of the Bayesian hypotheses must be included as structure tags and the distance error must be specified as its associated scalar value.  
 
## OPTIONAL INPUT KEYWORDS:
        /UNIT_PRIORS: If this keyword is set, all default priors are set to 1 (but they are still overrided by manual priors input with the keyword LN_PRIORS).
        /LNP_ONLY: If this keyword is set, only Bayesian probabilities will be calculated and returned.
        /NO_XYZ: If this keyword is set, the width of the spatial components of the multivariate Gaussian will be widened by a large   factor, so that the XYZ components are effectively ignored. This keyword must be used with extreme caution as it will generate a significant number of false-positives and confusion between the young associations.
        /USE_RV: Use any radial velocity values found in the stars_data input structure.
        /USE_DIST: Use any distance values found in the stars_data input structure.
        /USE_PLX: Use any parallax values found in the stars_data input structure.
        /USE_PSI: Use any psira, psidec values found in the stars_data input structure.
        /OVERRIDE_ERRORS: Do not exit program even when errors are encountered.
 
## OUTPUT:
This routine outputs a single IDL structure (or array of structures when many objects are analyzed at once), with the following tags:

        NAME: The name of the object (as taken from the input structure).
        ALL: A structure that contains the Bayesian probability (0 to 1) for each of the associations (as individual tags).
        METRICS: A structure that contains the performance metrics associated with the global Bayesian probability of this target.
                This sub-structure contains the following tags:
                TPR: Rate of true positives expected in a sample of objects that have a Bayesian membership probability at least as large as that of the target.
                FPR: Rate of false positives (from the field) expected in a sample of objects that have a Bayesian membership probability at least as large as that of the target.
                PPV: Positive Predictive Value (sample contamination) expected in a sample of objects that have a Bayesian membership probability at least as large as that of the target.
        BESTYA_STR: A sub-structure similar to those described above for the most probable young association (ignoring the field possibility).
        YA_PROB: The Bayesian probability (0 to 1) that this object belongs to any young association (i.e., excluding the field).
        LIST_PROB_YAS: A list of young associations with at least 5% Bayesian probability. Their relative probabilities (%) are specified between parentheses.
        BEST_HYP: Most probable Bayesian hypothesis (including the field)
        BEST_YA: Most probable single young association.
        [ASSOCIATION_1]: Sub-structure containing the relevant details for assiciation [ASSOCIATION_1].
        [ASSOCIATION_2]: (...)
        (...)
        [ASSOCIATION_N] - (...)

These per-association sub-structures contain the following tags:

        HYPOTHESIS: Name of the association.
        PROB: Bayesian probability (0 to 1).
        D_OPT: Optimal distance (pc) that maximizes the Bayesian likelihood for this hypothesis.
        RV_OPT: Optimal radial velocity (km/s) that maximizes the Bayesian likelihood for this hypothesis.
        ED_OPT: Error on the optimal distance (pc), which approximates the 68% width of how the likelihood varies with distance.
        ERV_OPT: Error on the optimal radial velocity (km/s), which approximates the 68% width of how the likelihood varies with radial velocity.
        XYZUVW: 6-dimensional array containing the XYZ and UVW position of the star at the measured radial velocity and/or distance, or the optimal radial velocity and/or distance when the first are not available (units of pc and km/s).
        EXYZUVW: Errors on XYZUVW (units of pc and km/s).
        XYZ_SEP: Separation between the optimal or measured XYZ position of the star and the center of the multivariate Gaussian model of this Bayesian hypothesis (pc).
        UVW_SEP: Separation between the optimal or measured UVW position of the star and the center of the multivariate Gaussian model of this Bayesian hypothesis (km/s).
        XYZ_SEP: N-sigma separation between the optimal or measured XYZ position of the star and the multivariate Gaussian model of this Bayesian hypothesis (no units).
        UVW_SEP: N-sigma separation between the optimal or measured UVW position of the star and the multivariate Gaussian model of this Bayesian hypothesis (no units).
        MAHALANOBIS: Mahalanobis distance between the optimal or measured XYZUVW position of the star and the multivariate Gaussian model. A Mahalanobis distance is a generalization of a 6D N-sigma distance that accounts for covariances.
