# (Transit) Clairvoyance
## Predicting multiple-planet systems for TESS

In 2017, TESS (Transiting Exoplanet Survey Satellite) will launch, marking a transition from the wide-apertured, long-staring capabilities of Kepler to the sweeping gaze of a new mission able to capture much more of the sky, at the expense of resolution and the chances of longer-period planet follow-up.

Transit Clairvoyance uses Artificial Neural Networks (ANNs) to predict the most likely short period transiters to have additional transiters, thereby doubling the discovery yield of a mock transit follow-up survey. The training and cross-validation were done by splitting the Kepler exoplanet data archive (http://exoplanetarchive.ipac.caltech.edu/, accessed May 17, 2016) and the features selected for an initial 2-feature ANN were the planetary radius and orbital period, while a binary inner multiplicity flag (ie. 0 for no companion planet with orbital period P < 13.7 days in the system; 1 otherwise) was added for a 3-feature ANN that we eventually combined with the 2-feature ANN into a hybrid. From the 2- and 3-feature ANNs, class probabilities were generated to describe how likely a system was to have exoplanets with P > 13.7. For a more detailed description of how these probabilities were generated and why 13.7 days was chosen as a threshold, see Kipping & Lam 2016. 

The 2- and 3-feature ANN results are displayed in the correspondingly named tables. In both, the columns are, in order:
- Log10( Radius [Earth radii] )
- Log10( Period of maximum sized planets )
- Probability of system having an outer transiter (0-1)

## Hybrid ANN
As mentioned above, the 3-feature ANN was trained on a binary inner multiplicity flag in addition to planetary radius and orbital period. Put together, the output of this and the 2-feature ANN is:
```
P = (1 - M_inner) * P_ANN2 + M_inner * P_ANN3
```
where P is the probability of a system having an outer transiter, 
```
M_inner = min(N_inner, 1)
```
and N_inner is the number of inner transiters in a system. In this way, the nodes of the hidden layer of ANN2 communicate with the first two features, while those of the hidden layer of ANN3 communicate with the multiplicity flag. 

[Click here for a look at the hybrid ANN architecture, from Kipping & Lam 2016.](HybridANN.png)

## Usage
[Clairvoyance](clairvoyance.py) is a simple 2-D interpolant that takes in the number of planets in a system with period less than 13.7 days, as well as the maximum radius amongst them (in Earth radii) and orbital period of the planet with maximum radius (in Earth days). Suppose you have just detected a system of N transiting planets with periods less than 13.7 days. In order to predict the probability of additional transiters in this system with period greater than 13.7 days, clone or download this repository. On a command line, type: python clairvoyance.py -n=N -r=R -p=P, where R is the maximum radius amongst the detected inner transiters and P is the orbital period of the planet with the greatest radius amongst them.

For example, consider a 2-inner-planet system, home to Planet 1, with radius of 4.214 Earth radii and orbital period of 0.878 days, and Planet 2, with radius of 2.493 Earth radii and orbital period of 5.682 days. After cloning Clairvoyance, since Planet 1 has the greater radius, its features will be the ones entered into the command line:
```
python clairvoyance.py -n=2 -r=4.214 -p=0.878
>> Clairvoyance predicts a 0.088662 percent probability of additional transiting planets with P > 13.7 days.
```

[Click here](hybrid_ann.py) for the interpolant code. 

