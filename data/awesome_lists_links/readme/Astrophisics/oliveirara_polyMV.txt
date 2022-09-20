# polyMV

<a href="http://ascl.net/2007.009"><img src="https://img.shields.io/badge/ascl-2007.009-blue.svg?colorB=262255" alt="ascl:2007.009" /></a> [![GitHub license](https://img.shields.io/github/license/oliveirara/polyMV)](https://github.com/oliveirara/polyMV/blob/master/LICENSE)

`polyMV` is a _Python_ package that converts multipolar coefficients (`alms` in `healpix` order) into Multipole Vectors (MVs) and also Fréchet Vectors (FVs) given a specific multipole.

Any publications making use of `polyMV` should cite this paper: R. A. Oliveira, T. S. Pereira, and M. Quartin, **CMB statistical isotropy confirmation at all scales using multipole vectors**, [Phys. Dark Univ. 30 (2020) 100608](https://doi.org/10.1016/j.dark.2020.100608) ([arXiv:1812.02654 [astro-ph.CO]](https://arxiv.org/abs/1812.02654)).

Checkout MVs and FVs from Planck 2015 and 2018 temperature maps in [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3866410.svg)](https://doi.org/10.5281/zenodo.3866410).

## Instalation

`polyMV` uses `MPSolve`. [MPSolve](https://github.com/robol/MPSolve) is a C package that finds roots of polynomials with high speed and precision. Before installing `polyMV`, compile and install `MPSolve`:

### MPSolve part:

1. Download `MPSolve` from source (make sure you have all dependencies):
    
    ```bash
    git clone https://github.com/robol/MPSolve.git
    ```


2. In `MPSolve` folder, run:
    
    ```bash
    bash autogen.sh
    ```


3. Configure:

    ```bash
    ./configure --prefix=<path-to-installation-folder>
    ```


4. Compile in parallel (it's faster):

    ```bash
    make -j 4
    ```
   
   
   In this case, the compilation will run in 4 threads.
   
5. Install `MPSolve`:

    ```bash
    make install
    ```
    
    
### polyMV part:

1. Clone this repository:

    ```bash
    git clone https://github.com/oliveirara/polyMV.git
    ```


2. Inside `src` folder, open `mpsolve.py` and replace the path for "libmps.so.3" on line 7:

    - on macOS:

    ```python
    _mps = ctypes.CDLL("libmps.so.3") -> _mps = ctypes.CDLL("<path-to-installation-folder>/MPSolve/lib/libmps.3.dylib")
    ```

    - on Linux:

    ```python
    _mps = ctypes.CDLL("libmps.so.3") -> _mps = ctypes.CDLL("<path-to-installation-folder>/MPSolve/lib/libmps.so.3")
    ```


`polyMV` is implemented to obtain fast roots of polynomials with precision up to 8 digits (about 0.02" on multipole scales). If you need more precision you should change in `mpsolve.py` file:

- Line 88:

```python
_mps.mps_context_set_output_prec(self._c_ctx, ctypes.c_long(53)) -> _mps.mps_context_set_output_prec(self._c_ctx, ctypes.c_long(XX))
```

where XX is the number of bits, not decimals.

- Line 89:

```python
Goal.MPS_OUTPUT_GOAL_ISOLATE -> Goal.MPS_OUTPUT_GOAL_APPROXIMATE
```


3. Install:

    ```bash
    pip install .
    ``` 


You also can add the flag `--user` to install locally.

## Notebooks:

In `notebooks` folder you will find some examples of how to use `polyMV`.

- - -

This work was funded by Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq), Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) and Fundação Araucária (PBA-2016).
