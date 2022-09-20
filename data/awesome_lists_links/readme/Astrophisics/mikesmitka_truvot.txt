This package contains the software required to perform the UVOT grism true background 
technique described in Smitka et al. 2015.  A working example using the data of
SN 2011fe is provided in the 'example' directory.



##CONTENTS:
  software/interpolate_function.pro
      A function used to interpolate a spectrum on to the wavelength scale of
      another spectrum.

  software/interpolate_function_errors.pro
      A function used to interpolate a spectrum on to the wavelength scale of
      another spectrum while retaining the flux errors.

  software/master_shift.pro
      A program used to align the wavelength scales of extracted spectra prior 
      to coadding.

  software/optimal_shift_time.pro
      The program used to 1) translate grism template images to overlay grism data
      images, 2) flux scale template images to match data images, 3) coadd template
      image snapshots for making a master template [coadd and ext2 flags must be set].

  software/required_packages.txt 
      A list of IDL and python packages required to execute the code in this
      repository.  

  software/spectra_avg.pro
      A program used to coadd extracted spectra from individual snapshots into a 
      single spectrum.
  
  example/00032094004 
      The directory containing the SN data grism images.

  example/00032094018
      The directory containing the template data grism images.

  example/idl_script1.pro 
      The first script to be executed in the example. Explanations of each line
      of code are provided as comments.  This script merely executes some of the IDL 
      programs in the software/ directory.  

  example/uvotpy_script.py
      The second script to be executed in the example.   Explanations of each line
      of code are provided as comments.  This script merely executes the UVOTPY 
      software (installed separately) in a way that utilizes the template image
      as the background.

  example/idl_script2.pro
      The third and final script to be executed in the example.  Explanations of
      each line of code are provided as comments.  This script merely executes some 
      of the IDL programs in the software/ directory.


##INSTALLATION:
  1. You should have working installations of UVOTPY and IDL before beginning.
  2. Download the TRUVOT software to a directory of your liking.
  3. Set your IDL path to recognize the programs in the software directory of 
     this package.
       e.g. for a tcshell in your .tcshrc file set:
           setenv IDL_PATH \+$IDL_DIR/lib:\+$IDL_DIR/user_contrib:\+/Users/username/truvot/software


##PROCEDURE:
  The easiest way to learn the process is to follow along line-by-line in the 3
  scripts.  The scripts must be executed in the order shown below because the output of
  each script is used in the subsequent script.  'Human-speak' explanations of the calling
  sequences are provided as comments in each script and in the program headers.  

  1. Navigate to the example directory (truvot/example)
  2. launch IDL and execute idl_script1.pro 
    -idl
    -IDL> idl_script1
  3. Launch ipython and execute uvotpy_script.py
    -heainit
    -ipy --pylab
    -ipy> run uvotpy_script.py
  4. launch IDL and execute idl_script2.pro
    -idl
    -IDL> idl_script2
  5. Check that your final output spectrum (in the file 
     truvot/example/00032094004/uvot/image/sw00032094004ugu_1ord_final.dat)
     is identical to what I got when I ran this procedure ( in the file 
     truvot/example/example_final_spectrum.dat).  If your output agrees
     with the reference case then you are good to go.

##VERSION:
Created by Mike Smitka, August 24, 2015.

##LICENSE
Please see the LICENSE file for details concerning the modification, 
distribution and citation of this software. 

##CONTACT:
mikesmitka@gmail.com
  I would like to know about your experience using the TRUVOT software.  I
  am happy to be of assistance to ensure a successful implementation.
  Please use the word 'truvot' in the subject of emails to assist in sorting and
  archiving of emails.
