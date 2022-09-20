Exoplanet population inference
==============================

This repository contains the code and text for the paper [Exoplanet population
inference and the abundance of Earth analogs from noisy, incomplete catalogs](
http://arxiv.org/abs/1406.3020)
by Daniel Foreman-Mackey, David W. Hogg, and Timothy D. Morton and submitted
to ApJ.

The code lives in the `code` directory and the LaTeX source code for the paper
is in `document`.

**Code**

The meat of the probabilistic model is implemented in `population.py`. Then
there are a set of scripts that you can use to generate the figures from the
paper. You should look at the docstrings for details but the summary is:

* `simulate.py` generates synthetic catalogs from known occurrence rate
  density functions,
* `main.py` does the MCMC analysis on either real or simulated catalogs, and
* `results.py` analyzes the results of the MCMC, makes some figures, and thins
  the chain to the published form.

Results
-------

Our simulated catalogs and results are [available online on figshare](
http://dx.doi.org/10.6084/m9.figshare.1051864).

Attribution
-----------

This code is associated with and written specifically for [our
paper](http://arxiv.org/abs/1406.3020). If you make any use of it,
please cite:

```
@article{exopop,
   author = {{Foreman-Mackey}, D. and {Hogg}, D.~W. and {Morton}, T.~D.},
    title = {Exoplanet population inference and the abundance of
             Earth analogs from noisy, incomplete catalogs},
  journal = {ArXiv --- submitted to ApJ},
     year = 2014,
   eprint = {1406.3020}
}
```

License
-------

Copyright 2014 Daniel Foreman-Mackey

The code in this repository is made available under the terms of the MIT
License. For details, see the LICENSE file.
