# Bayesian SZNet: Bayesian deep learning to predict redshift with uncertainty

Bayesian SZNet predicts spectroscopic redshift through use of a Bayesian convolutional network.
It uses Monte Carlo dropout to associate predictions with predictive uncertainties,
allowing the user to determine unusual or problematic spectra for visual inspection
and thresholding to balance between the number of incorrect redshift predictions and coverage.

We use Julia 1.5.1 and Python 3.9.5.
