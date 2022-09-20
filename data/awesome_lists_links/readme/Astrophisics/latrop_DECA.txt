### Instructions on DECA Installation:

You need to have:
1) Python >= 2.6 (but not Python3)

2) Numpy, IPython, Matplotlib and SciPy (you can install all of them at once following instructions at http://www.scipy.org/install.html. 

3) SExtractor. You can find it here: http://www.astromatic.net/software/sextractor

4) GALFIT >=3.0. WARNING: If you have downloaded a binary from the GALFIT site (http://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html) there would be some problems with running it inside the DECA code. The problem is not yet found but this can happen. Therefore we strongly recommend you to ask Dr. Peng to send you a source copy of GALFIT so you could compile it under your OS. He would be glad to do it for you.

5) SAOImage DS9.It can be easily downloaded from here: http://hea-www.harvard.edu/RD/ds9/site/Download.html. If you use Ubuntu, you need to do also this command:sudo apt-get install libxss1. In SuSe I had an error with ds9 so I needed to type in the terminal: $ XPA_METHOD=local; export XPA_METHOD (you can add this line in your bash-profile).

6) Some additional Python modules: PyFITS (http://www.stsci.edu/institute/software hardware/pyfits
) and fpdf (https://code.google.com/p/pyfpdf/). There would not be a problem to install them two.

7) IRAF with STSDAS. There are inofficial instructions in youtube how to do it (see http://www.youtube.com/watch?v=BtTr_F08y7o and http://www.youtube.com/watch?v=J4fkRrWHCFY)

Note here that binaries of SExtractor, GALFIT and ds9 must be placed, for example, in /usr/local/bin or smth. like that so they could be called by $ sex [args], $ galfit [args] and ds9 [args] from any directory in home/. Or you can create symbolic links to them.

After installing all the programs and modules above you can run DECA from any directory (do not forget to create input files) just by $ python [path_to_DECA] [args], where [path_to_DECA]/deca.py is the full path to the directory of downloaded DECA version with the main file deca.py. You can read about [args] for DECA in manual.pdf. Also you may export path to DECA directory by adding the line to your .bashrc:
	export PATH=[path_to_DECA]:$PATH  (e.g. PATH=/home/mosenkov/diser/DECA_1.0.5:$PATH)
	or, export PATH=$PATH:[path_to_DECA]

For tcsh or csh shell enter:
	set PATH = ($PATH [path_to_DECA])
	or, setenv PATH $PATH:[path_to_DECA]

Make the file deca.py in the DECA directory executable by enetering: $ chmod +x deca.py.
Do not forget to close the terminal and then restart it. Now you can run DECA just by: $ deca.py [args].

Remember that DECA uses IRAF so you always need to have a login.cl file in the directory where you run DECA (mkiraf => xgterm). Always change the line with `set imdir' (the path in `set imdir' must be the same as in `set home'), for example:

set home = "/home/mosenkov/GALAXY/"
set imdir = "/home/mosenkov/GALAXY/"

Now you are ready to work with DECA!

Please, read the description of DECA code in manual.pdf and article.pdf. The Installation instructions can be found in manual.pdf. The bash script DECA_INSTALL.sh for Ubuntu 32/64 can be executed by: $ sh DECA_INSTALL.sh, where you can find the same instructions as have been described here. 
