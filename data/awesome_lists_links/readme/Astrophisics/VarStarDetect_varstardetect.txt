Varstar Detect
==============

Python package optimized for variable star detection in TESS Secctor 1
data.

 Initialization:
--------------------

To install repository from PyPi: `pip install varstardetect`

 Looking for stellar variability:
-------------------------------------

Use the `amplitude_test()` function, with the following documentation,
to determine objects which are variable.

    amplitude_test(min, max, amp)

      """
                  amplitude_test DOCUMENTATION
    ---------------------------------------------------------------------
    Detects variable stars with amplitude higher than threshold.
    ---------------------------------------------------------------------
    INPUTS:     - min: lower star search (TESS) range delimiter
                - max: higher star search (TESS) range delimiter
                - amp: amplitude threshold
                - dir: directory with TESS sector 1 files csv. You can
                  download from
                  https://tess.mit.edu/observations/target-lists/
            -------------------------------------------------------------
    OUTPUTS:    - candidates: 1D numpy array with variable candidate
                target IDs
                - chis: 1D numpy array with the chi^2 parameter of each
                approximation.
                - degree: 1D numpy array with the degree of the optimal
                degree of the approximation.
                - periods: 1D numpy array with the period of each
                approimating function.
                - period_errors: 1D numpy array with the period
                uncertainty for each candidate.
                - amplitudes: 1D numpy array with the amplitude of each
                approximation.
                - amplitude_errors: 1D numpy array with the uncertainty
                of the amplitude of each candidate.
    ----------------------------------------------------------------------
    PROCEDURE:
                1. Calculates amplitude for an observed star.
                2. Calculates if amplitude is bigger than threshold.
                3. Returns candidates and their characteristics.
    ----------------------------------------------------------------------
      """

Background:
-----------

The function uses several numerical and statistical methods to filter
and interpret the data obtained form TESS, providing the characteristics
of each star through phenomenological analysis of the lightcurve, given
that it has passed the amplitude test.

DISCLAIMER:
-----------

This is a Beta state of the program. It is unstable therefore itcan have 
bugs. It has not been optimized correctly yet.
