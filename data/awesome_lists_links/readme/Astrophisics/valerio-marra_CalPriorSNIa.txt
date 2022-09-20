# **CalPriorSNIa**
## Effective calibration prior on the absolute magnitude of Type Ia supernovae

**priorMB.nb** quickly computes the effective calibration prior on the absolute magnitude M_B of Type Ia supernovae that corresponds to a given determination of H_0. 
See Camarena & Marra [arXiv:1906.11814](https://arxiv.org/abs/1906.11814) and [arXiv:2101.08641](https://arxiv.org/abs/2101.08641) for more details.

Have fun,
[David](http://inspirehep.net/author/profile/D.Camarena.1) & [Valerio](http://inspirehep.net/author/profile/V.Marra.1)


### Calibration priors

The latest H_0 determination by SH0ES is (Riess et al. [arXiv:2112.04510](https://arxiv.org/abs/2112.04510)):    
`H_0 = 73.04 ± 1.04 km/s/Mpc`

The corresponding calibration priors depend on the supernova dataset that one wants to use for the cosmological analysis:

DES-SN3YR dataset ([Brout et al 2018](https://arxiv.org/abs/1811.02377)) ==>
`M_B = -19.2432 ± 0.0245 mag`

Pantheon dataset ([Scolnic et al 2018](https://arxiv.org/abs/1710.00845)) ==>
`M_B = -19.2478 ± 0.0294 mag`

Pantheon+ dataset ([Scolnic et al 2021](https://arxiv.org/abs/2112.03863)) ==>
`coming...`    
see [github.com/PantheonPlusSH0ES](https://github.com/PantheonPlusSH0ES/DataRelease) for the CosmoSIS likelihood which includes the full covariance matrix between supernovae and calibrators.

### Statistical analysis

When performing cosmological inference, instead of using the prior on H_0:

<a href="https://www.codecogs.com/eqnedit.php?latex=\chi^2_{H_0}&space;=&space;\frac{\big&space;(H_0-H^{\rm&space;SH0ES}_0&space;\big)^2}{\sigma_{H_0}^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\chi^2_{H_0}&space;=&space;\frac{\big&space;(H_0-H^{\rm&space;SH0ES}_0&space;\big)^2}{\sigma_{H_0}^2}" title="\chi^2_{H_0} = \frac{\big (H_0-H^{\rm SH0ES}_0 \big)^2}{\sigma_{H_0}^2}" /></a>,

use the prior on M_B:

<a href="https://www.codecogs.com/eqnedit.php?latex=\chi^2_{M_B}&space;=&space;\frac{\big&space;(M_B-M_B^{\rm&space;SH0ES}&space;\big&space;)^2}{\sigma_{M_B}^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\chi^2_{M_B}&space;=&space;\frac{\big&space;(M_B-M_B^{\rm&space;SH0ES}&space;\big&space;)^2}{\sigma_{M_B}^2}" title="\chi^2_{M_B} = \frac{\big (M_B-M_B^{\rm SH0ES} \big )^2}{\sigma_{M_B}^2}" /></a>.

The prior on the supernova absolute magnitude M_B is to be preferred to the prior on H_0 for the following three reasons:<br/>
i) one avoids potential double counting of low-redshift supernovae,<br/>
ii) one avoids cosmography, in particular fixing the deceleration parameter to the standard model value of q_0=-0.55 (see [arXiv:2112.04510](https://arxiv.org/abs/2112.04510) for details),<br/>
iii) one includes in the analysis the fact that M_B is constrained by local calibration, an information which would otherwise be neglected in the analysis.


### Monte Python likelihood

The directory [montepython_prior](https://github.com/valerio-marra/CalPriorSNIa/tree/master/montepython_prior) contains the M_B likelihood for [Monte Python](https://github.com/brinckmann/montepython_public) together with a parameter file example.  Monte Python v3.5 already includes this likelihood.

### Credits

You can use CalPriorSNIa, or part of it, freely, provided that in your publications you acknowledge its use and cite the papers Camarena & Marra [arXiv:1906.11814](https://arxiv.org/abs/1906.11814) and [arXiv:2101.08641](https://arxiv.org/abs/2101.08641).
If using the provided supernova data, please cite the corresponding paper.
