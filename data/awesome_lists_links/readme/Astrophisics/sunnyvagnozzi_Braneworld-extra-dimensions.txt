# Braneworld-extra-dimensions

This repository contains a set of results and tools to reproduce the results from:

L. Visinelli, N. Bolis, S. Vagnozzi, [Brane-world extra dimensions in light of GW170817](https://inspirehep.net/record/1636969), [Phys. Rev. D97 (2018) 064039](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.97.064039) [[arXiv: 1711.06628]](https://arxiv.org/abs/1711.06628) (VBV17)

where constraints on the size of the AdS<sub>5</sub> radius of curvature within the [Randall-Sundrum brane-world model](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.83.4690) are placed in light of the near-simultaneous detection of the gravitational wave event GW170817 and its optical counterpart, the short γ-ray burst event GRB170817A ([Astrophys.J. 848 (2017) no.2, L13](http://iopscience.iop.org/article/10.3847/2041-8213/aa920c/meta)). The content of the folders is described below.

## montepython

This folder contains a patch to the [Montepython](https://github.com/baudren/montepython_public) cosmological MCMC sampler to sample the posterior distribution of the 4-dimensional parameter space in VBV17 and obtain constraints on the parameters. You should therefore first of all download and install [Montepython](https://github.com/baudren/montepython_public) following the documentation provided.

The *[GW170817_braneworld/](https://github.com/sunnyvagnozzi/Braneworld-extra-dimensions/tree/master/montepython/GW170817_braneworld)* folder implements the likelihood (Eq.(19) in VBV17), hence the name of the experiment you want to use in your run is "GW170817_braneworld" as per Montepython's philosophy. You should copy this folder into your montepython/montepython/likelihoods folder.

You should also copy the contents of the *[covmat/](https://github.com/sunnyvagnozzi/Braneworld-extra-dimensions/tree/master/montepython/covmat)* folder into your montepython/covmat folder (assuming you want to supply an input covariance matrix to your runs, which is recommended).

The input .param file to do the MCMC run is *braneworld.param*. To run the MCMC you then want to do something like (customize your arguments as you prefer, see [Montepython's documentation](http://monte-python.readthedocs.io/en/latest/)): 

    $ python montepython/MontePython.py run -p braneworld.param -o chains/braneworld -N 100000 --chain-num 1 --update 300 -c covmat/braneworld.covmat [optional arguments]

The supplied *braneworld.sl* file is a batch file which can be used to run the MCMC on the NERSC supercomputer [Cori](http://www.nersc.gov/users/computational-systems/cori/). To submit, type:

    $ sbatch braneworld.sl

The script runs 8 chains in parallel on 1 node, using 4 cores/chain (Cori has 32 cores per node) for 40 minutes (which is more than enough to get a convergence R-1 way better than 0.01. If you want to run on the NERSC supercomputer [Edison](http://www.nersc.gov/users/computational-systems/edison/) we recommend instead running 6 chains (i.e. change "-n 8" to "-n 6" in braneworld.sl). You can also use hyperthreading but you don't gain much in terms of running time.

## plots

This folder contains the scripts used the produce the 3 figures in VBV17. To run, simply type (directly from the *[plots/](https://github.com/sunnyvagnozzi/Braneworld-extra-dimensions/tree/master/plots)* folder):

    $ python file_name.py
    
where file_name=fig1, fig2, or fig3. This will produce fig1.pdf, fig2.pdf, and fig3.pdf directly in the *[plots/](https://github.com/sunnyvagnozzi/Braneworld-extra-dimensions/tree/master/plots)* folder. Make sure you have installed the [getdist](https://getdist.readthedocs.io/en/latest/) Python package.

## results

This folder contains the results of the MCMC analysis used to write VBV17. The MCMC chains we generated are contained in the folder *[chains/](https://github.com/sunnyvagnozzi/Braneworld-extra-dimensions/tree/master/results/chains)*.

The folder *[plot_data/](https://github.com/sunnyvagnozzi/Braneworld-extra-dimensions/tree/master/results/plot_data)* contains the output of running [getdist](https://getdist.readthedocs.io/en/latest/) on these chains, and is needed to produce [Fig.1](https://github.com/sunnyvagnozzi/Braneworld-extra-dimensions/blob/master/plots/fig1.pdf).

The files *braneworld.corr*, *braneworld.covmat*, *braneworld.likestats* and *braneworld.margestats* are standard files produced by getdist when analysing the chains. You will need *braneworld.corr* to produce [Fig.3](https://github.com/sunnyvagnozzi/Braneworld-extra-dimensions/blob/master/plots/fig3.pdf), whereas the numbers in *braneworld.margestats* are those we actually quote in VBV17 (68% and 95% CL upper limits on the AdS<sub>5</sub> radius of curvature).

# References

If you use these codes in your research, please cite the following paper (if you use *Montepython*, you should also cite the original work):

L. Visinelli, N. Bolis, S. Vagnozzi, [Brane-world extra dimensions in light of GW170817](https://inspirehep.net/record/1636969), [Phys. Rev. D97 (2018) 064039](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.97.064039) [[arXiv: 1711.06628]](https://arxiv.org/abs/1711.06628)

### Latex (EU) entry

    %\cite{Visinelli:2017bny}
    \bibitem{Visinelli:2017bny}
      L.~Visinelli, N.~Bolis and S.~Vagnozzi,
      %``Brane-world extra dimensions in light of GW170817,''
      Phys.\ Rev.\ D {\bf 97} (2018) 064039
      doi:10.1103/PhysRevD.97.064039
      [arXiv:1711.06628 [gr-qc]].
      %%CITATION = doi:10.1103/PhysRevD.97.064039;%%
      %3 citations counted in INSPIRE as of 01 Apr 2018

### Bibtex entry

    @article{Visinelli:2017bny,
          author         = "Visinelli, Luca and Bolis, Nadia and Vagnozzi, Sunny",
          title          = "{Brane-world extra dimensions in light of GW170817}",
          journal        = "Phys. Rev.",
          volume         = "D97",
          year           = "2018",
          pages          = "064039",
          doi            = "10.1103/PhysRevD.97.064039",
          eprint         = "1711.06628",
          archivePrefix  = "arXiv",
          primaryClass   = "gr-qc",
          SLACcitation   = "%%CITATION = ARXIV:1711.06628;%%"
    }

# Support

If you experience any problems with the codes, or to report bugs, please email [sunny.vagnozzi@fysik.su.se](mailto:sunny.vagnozzi@fysik.su.se) (unless you're a robot, of course)

************************************************************************************************

This research was supported by:

   <a href="http://www.okc.albanova.se/"><img src="http://www.okc.albanova.se/polopoly_fs/1.327382.1491483655!/image/image.jpg_gen/derivatives/logotype_h130/image.jpg"
height="100px"></a>
   <a href="https://www.su.se/"><img src="http://resources.mynewsdesk.com/image/upload/t_open_graph_image/ayjgabd4qxqbpj4pu4nl.jpg"
height="100px"></a>
      <a href="https://www.nordita.org/"><img src="https://yt3.ggpht.com/a-/AJLlDp3bQ-UG2qVRBjqfsEbsUaDs_fd8yBPkMnPCXg=s900-mo-c-c0xffffffff-rj-k-no"
height="100px"></a>
   <a href="http://www.ceico.cz/"><img src="https://academicpositions.eu/uploads/46e/083/46e083d07d2516e6b22c300bfe4731ac.jpeg" height="100px"></a>
   <a href="https://www.fzu.cz/"><img src="https://www.fzu.cz/sites/default/files/logo-FZU-velke_1000x600px.jpg" height="100px"></a>
