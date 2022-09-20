# TERRA #

A suite of codes to find extrasolar planets

## Dependencies ##

### Public Python modules ###
```
- NumPy (tested with v1.11.0)
- SciPy (tested with 0.18.0)
- matplotlib 
- pandas (v0.18)
- lmfit, for transit fitting ()
- batman
```

## Installation Instructions ##

1. clone the git repo. Will create a `</path/to/terra>` directory
2. cd into target directory
   
   ```
   /global/homes/p/petigura/code_carver/terra
   ```
   
3. Build the cython and fortran extensions with make. At NERSC, there was a little difficulty getting the fortran and c libraries to compile. Here's what I had to do:

   ```
   module load gcc
   # This will compile the cython extensions but the -dynamic lookup won't pass to the gfort
   LDFLAGS="-L/global/homes/p/petigura/anaconda/lib" SOFLAGS="-fPIC -shared" make
   # Running it a second time will get the fortran code compiled
   make
   
   # the last compiled fortran code should look something like:
   /opt/gcc/4.9.2/bin/gfortran -Wall -g -Wall -g -shared
   ```

4. Add `</path/to/terra>` to $PYTHONPATH
5. Add `</path/to/terra> /bin/` to $PATH
6. Test that everything is working by running the following test

   ```
   python -c "from terra import terra; terra.test_terra()"
   ```
   
   
