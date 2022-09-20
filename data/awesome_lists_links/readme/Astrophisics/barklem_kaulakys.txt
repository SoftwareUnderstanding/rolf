# kaulakys

kaulakys.pro - library of codes to calculate cross sections and rate coefficients for inelastic collisions with hydrogen atoms according to the free electron model of Kaulakys (1986, 1991).  Written in IDL.

Two examples of how one might use the code are provided in Li_kaul.pro and Na_kaul.pro and the results I obtained are in kaul_Li.txt and kaul_Na.txt.  Various other tests in test_kaulakys.pro and test.pro.

The code depends on mswavef/mswavef.pro and libpub/integ_rate.pro

The code could be easily adapted to collisions with perturbers other than hydrogen atoms by providing the appropriate scattering amplitudes for e+perturber scattering.

Please cite the DOI if you use the code in research:
[![DOI](https://zenodo.org/badge/21607/barklem/kaulakys.svg)](https://zenodo.org/badge/latestdoi/21607/barklem/kaulakys)

References:

Kaulakys B (1991) Free electron model for collisional angular momentum mixing of high Rydberg atoms. J Phys B At 24(5):L127–L132, DOI 10.1088/0953-4075/24/5/004

Kaulakys BP (1986) Free electron model for inelastic collisions between neutral atomic particles and Rydberg atoms. Journal of Experimental and Theoretical Physics 91:391– 403

Bug reports: paul.barklem@physics.uu.se
