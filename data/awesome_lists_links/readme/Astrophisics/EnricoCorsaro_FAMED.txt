# FAMED - Fast and AutoMated pEak bagging with Diamonds
<p align="center">
<a href="https://github.com/EnricoCorsaro/FAMED"><img src="https://img.shields.io/badge/GitHub-FAMED-yellow"/></a>
<a href="https://github.com/EnricoCorsaro/FAMED/blob/master/LICENSE.txt"><img src="https://img.shields.io/badge/license-MIT-blue"/></a>
<a href="https://www.aanda.org/articles/aa/abs/2020/08/aa37930-20/aa37930-20.html"><img src="https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F202037930-blueviolet"/></a>
<a href="https://ascl.net/2006.021"><img src="https://img.shields.io/badge/ASCL-2006.021-red"/></a>
<a href='https://famed.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/famed/badge/?version=latest' alt='Documentation Status' /></a>
<a href="https://github.com/EnricoCorsaro/FAMED/issues"><img src="https://img.shields.io/github/issues-closed/EnricoCorsaro/FAMED"/></a>
<img width="400" src="./docs/figures/FAMED_LOGO_WHITE.jpeg"/>
</p>

### Author
- [Enrico Corsaro](mailto:enrico.corsaro@inaf.it)

### Python version development
- [Jean McKeever](mailto:jm@mira.org)
- [James Kuszlewicz](mailto:kuszlewicz@mps.mpg.de)


### Short description
<div align="justify">
The <b>FAMED</b> (<b>F</b>ast and <b>A</b>uto-<b>M</b>ated p<b>E</b>ak bagging with <b>D</b>iamonds) pipeline is a multi-platform parallelized software to perform an automated extraction and mode identification of oscillation frequencies for solar-like pulsators. This pipeline is based on the free code DIAMONDS for Bayesian parameter estimation and model comparison by means of the nested sampling Monte Carlo (NSMC) algorithm. The pipeline can be applied to a large variety of stars, ranging from hot F-type main sequence, up to stars evolving along the red giant branch, settled into the core-Helium-burning main sequence, and even evolved beyond towards the early asymptotic giant branch.
The pipeline is organized in separate modules, each one performing different tasks in a different level of detail. The current version of FAMED includes two out of four modules, which are named GLOBAL and CHUNK. These two first modules are available in <code class="docutils literal notranslate"><span class="pre">IDL</span></code> version, and are being developed in <code class="docutils literal notranslate"><span class="pre">Python</span></code> too. The pipeline requires some system prerequisites and a dedicated installation procedure which can be found in the documentation below.
</div>

### Documentation
Please make sure you read the documentation at [famed.readthedocs.io](http://famed.readthedocs.io/) before installing and using the code.
