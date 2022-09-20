# ExoPriors
Calculate log-likelihood penalties to account for observational bias in exoplanet detection.

ExoPriors is a Python code that accounts for observational bias (geometric and signal-to-noise ratio detection bias) of transiting exoplanets by calculating a log-likelihood penalty for an input set of transit parameters.

ExoPriors calculates this log-likelihood penalty in one of four user-specified cases (for full explanation of these cases, see Kipping & Sandford 2016):

1. 'geometric_only': In this case, only the geometric bias is included, not the SNR detection bias. This is equivalent to assuming infinite SNR.
2. 'nongrazing_only': In this case, both types of bias are included, and grazing transit events are NOT considered detections.
3. 'general': In this case, both types of bias are included, and detected transits may be grazing or non-grazing.
4. 'occultation': In this case, both types of bias are included, and the event is an occultation with depth 'docc' rather than a transit.

In order to calculate the appropriate log-likelihood penalty for a transit/occultation event, call the function:

transit_LL_penalty(case=[one of the four options outlined above], appropriate kwargs) 

where the kwargs are:

    - per (REQUIRED) = orbital period of transiting planet
    - rhostar (REQUIRED) = density of host star
    - omega (REQUIRED) = argument of periapsis
    - e (REQUIRED) = eccentricity
    - ror (REQUIRED UNLESS case=='geometric_only') = ratio of planet radius to stellar radius
    - b (REQUIRED UNLESS case=='geometric_only') = impact parameter
    - occdepth (REQUIRED IF case=='occultation') = occultation depth 
    - blend (OPTIONAL) = blend factor = (target_star_flux + blended_source_flux)/target_star_flux (see equation 5 of Kipping & Tinetti 2010, MNRAS, 407, 2589)

You should not need to call any of the other functions defined in ExoPriors directly.
