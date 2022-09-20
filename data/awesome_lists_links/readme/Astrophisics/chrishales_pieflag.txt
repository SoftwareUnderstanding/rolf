pieflag
=======

![logo](./pieflag.jpg)

[CASA](http://casa.nrao.edu/) task to flag bad data by drawing upon statistics from reference channels in bandpass-calibrated data.

Let's face it, flagging is laborious and boring. You want a simple and effective tool to fight rfi like a boss. pieflag operates on a simple philosophy: you need to do some very basic pre-processing of your data so that the code can do its job robustly using a minimal set of assumptions.

pieflag works by comparing visibility amplitudes in each frequency channel to a 'reference' channel that is rfi-free, or manually ensured to be rfi-free. To operate effectively, pieflag must be supplied with bandpass-calibrated data (preliminary gain-calibration is also preferable).

pieflag has two core modes of operation -- static and dynamic flagging -- with an additional extend mode. Which mode you choose is largely dictated by the type of data you have.

An extensive help file is included with pieflag that includes instructions for pre-processing your data and selecting the best modes of operation. You are strongly encouraged to read the full documentation before running pieflag.

Once you have carried out your pre-processing and selected the mode of operation, pieflag should work well 'out of the box' with its default parameters. By comparing to a clean reference channel, essentially all bad data will be identified and flagged by pieflag.

Starting with version 4.0, pieflag is capable of parallel processing within the CASA MPI environment. Note that if you are using a CASA version from series 4.3 or earlier, you are limited to using pieflag version [3.2](https://github.com/chrishales/pieflag/releases/v3.2).

Latest version: 4.4 ([download here](https://github.com/chrishales/pieflag/releases/v4.4))

Tested with CASA 4.7.0 using Jansky VLA data

pieflag originally written by Enno Middelberg 2005-2006 (Reference: [E. Middelberg, 2006, PASA, 23, 64](http://arxiv.org/abs/astro-ph/0603216)). Starting with Version 2.0, pieflag has been rewritten for use in CASA and updated to include wideband and SEFD effects by Christopher A. Hales (Reference: [C. A. Hales, E. Middelberg, 2014, Astrophysics Source Code Library, 1408.14](http://adsabs.harvard.edu/abs/2014ascl.soft08014H)).

pieflag is released under a BSD 3-Clause License (open source, commercially useable); refer to LICENSE for details.

pieflag logo created by Chris Hales and the amazing graphic designer [Jasmin McDonald](http://www.theloop.com.au/JasminMcDonald/portfolio).

Correspondence regarding pieflag is always welcome.

Installation
======

Download the latest release from [here](https://github.com/chrishales/pieflag/releases/latest).

Place the source files into a directory containing your measurement set. Without changing directories, open CASA and type
```
os.system('buildmytasks')
```
then exit CASA. A number of files should have been produced, including ```mytasks.py```. Reopen CASA and type
```
execfile('mytasks.py')
```
To see the parameter listing, type
```
inp pieflag
```
For more details on how plot3d works, type
```
help pieflag
```
Now set some parameters and press go!

For a more permanent installation, place the source files into a dedicated pieflag code directory and perform the steps above. Then go to the hidden directory ```.casa``` which resides in your home directory and create a file called ```init.py```. If you are installing version 4.0 or higher, include the following line
```
execfile('/<path_to_pieflag_directory>/mytasks.py')
```
If you are installing version 3.2 or earlier, also add the following lines
```
os.environ['PYTHONPATH']='/<path_to_pieflag_directory>:'+os.environ['PYTHONPATH']
sys.path.append('/<path_to_pieflag_directory>')
```
pieflag will now be available when you open a fresh terminal and start CASA within any directory.

Acknowledging use of pieflag
======

We would appreciate your acknowledgement by citing [this ASCL entry](http://adsabs.harvard.edu/abs/2014ascl.soft08014H).