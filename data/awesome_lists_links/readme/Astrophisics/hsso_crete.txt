CRETE (Comet RadiativE Transfer and Excitation)
-------

CRETE is a 1D water excitation and radiation transfer code for
sub-millimeter wavelengths based on the `RATRAN` code [(Hogerheijde & van der Tak
2000)](http://adsabs.harvard.edu/abs/2000A%26A...362..697H).
The code considers rotational transitions of water molecules given
a Haser spherically symmetric distribution.

Installing the code
-------

From the top-level directory, run the `configure.sh` shell script to install
`crete`.

```
./configure.sh
```

This script is a simplified version of the original `RATRAN`'s `configure`
script that works for `bash` and `zsh` shells on Linux, and probably also
MacOS, using the `gfortran` compiler.

If you are using the C-shell or the improved version tcsh for interactive use,
you can instead run the csh script to install the program:

```
./configure
```

After running either of these scripts you can add the definition of the RATRAN
and RATRANRUN environment variables to initialize these variables for login
shells in the shell-specific files `~/.bashrc` or `~/.zshrc`:

```
export RATRAN=/path/to/crete
export PATH=$PATH:$RATRAN/bin
export RATRANRUN=$RATRAN/run
```

For `~/.cshrc`:

```
setenv RATRAN /path/to/crete
set path = ($RATRAN/bin $path)
setenv RATRANRUN $RATRAN/run
```

The IR radiation is specified as pumping coefficients in the file
`amc/getmatrix.f`.  One needs to use the `g_ir` array for ortho- or
para-water by uncommenting the corresponding definition from [Zakharov et al
2007](http://adsabs.harvard.edu/abs/2007A%26A...473..303Z). The model file
includes the heliocentric distance of the source to scale the IR pumping
coefficients.
