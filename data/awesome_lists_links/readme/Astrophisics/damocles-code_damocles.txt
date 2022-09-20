# Damocles

Damocles is a 3D Monte Carlo line transfer code designed to model the effects of dust on line emission an expanding atmosphere. It is a grid-based code with a wide variety of variable parameters. It is suitable for modelling the presence of dust in supernovae or supernovae remnants, stellar winds, planetary nebulae and AGN. For more details on how Damocles works, please see [here](https://github.com/damocles-code/damocles/wiki/About).

Please cite Bevan, A. & Barlow, M. J. 2016, 456, 1269 and Bevan, A. 2018, MNRAS, 480, 4659 in any work that uses this code.

## Quick-start guide
For full details on how to install and run Damocles, please see the online [wiki](https://github.com/damocles-code/damocles/wiki).

### Installation
Compile and install Damocles by typing:

``` 
make install
```

To additionally install the interactive version of Damocles developed by Maria Niculescu-Duvaz, type

```
make damocles-interactive
```

For further details, please see the [installation page](https://github.com/damocles-code/damocles/wiki/Installation).

### Set-up and run
All variables are specified in the input files with extension '.in' in the ```input``` folder. Adjust these variables as required and save the files. Run by typing ```damocles``` from the main directory. All output files are saved to the ```output``` folder.

For further details on running Damocles, please see the [Running Damocles](https://github.com/damocles-code/damocles/wiki/Running-Damocles) pages.

## License

Damocles is released under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version. This requires that any changes or improvements made to the program should also be made freely available.
