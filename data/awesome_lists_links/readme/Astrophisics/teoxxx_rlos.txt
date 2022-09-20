RLOS version 1.15.1, by T. Smponias, 2015-2018

This is  Relativistic Line Of Sight (RLOS), released under the LGPL-3.0-or-later licence. It is a program that produces synthetic, time-delayed images of model relativistic astrophysical systems. It crosses with parallel lines of sight the model system, resulting to the synthetic image. The geometry of the model system may be either input manually, for simpler geometries that are steady-state. Or RLOS may run on the data output of a time-dependent hydrocode, which is the case in this version of the software. More specifically, RLOS is currently optimized to run on the results of PLUTO, written by A. Mignone and his team. There are two main source code files in RLOS, for imaging on two different planes, called XZ and YZ respectively.

Project website: https://github.com/teoxxx/rlos

Language: IDL, or GDL


**************************************************************************************************************

What’s New

This version of RLOS is able to produce a special-relativistic, time-delayed image, of a model relativistic astrophysical system, such as a jet. 

RLOS supports data input from the PLUTO hydrocode (dbl or vtk formats). It may also be adapted to employ those same data formats from another hydrocode. A steady-state model setup is also included inert, and can be activated when necessary. 



Configuration options for RLOS

You can run RLOS 1.15.1 on a system configured in a number of ways. Regardless of how the system is configured, it will contain these elements:

.Data output from a hydrocode in dbl (preferably) or vtk format. Also, other accompanying files of the data output.

.Required software to load those data into RLOS. In this implementation, it consists of PLUTO's attached IDL suite of routines, pload.pro. The pload.pro routine will call further accompanying routines, which also need to be present. These are located in the PLUUTO distribuution, under ..../Tools/IDL/.

.The external parameter file of RLOS: rlos_params.txt.

.IDL or GDL installed.



System requirements

A modern PC should suffice in order to run RLOS, at limited grid resolution. In particular, RLOS should run on a system where a multiple quantity of RAM is available, as compared to the amount needed by the hydrocode producing the data. Please see the relevant scientific paper for more details.


RLOS documentation is updated for this release.


Known Issues

The following issues have now been resolved.

1 RLOS now may work at reduced grid resolution (as a quick preview of the result), using pload's shrink factor. 
2 The frequency shift effect may now be either used explicitly (FS on) or implicitly (as part of the DB formula).


Enhancements

The following enhancements are now available:

1 RLOS may or may not display debug comments during execution, depending on a setting in the rlos_params.txt file.
2 Parameter values from the input parameter text file now appear automatically as annotations on the final synthetic image.
3 Datapath string contains path to data. pload.pro, PLUTO's IDL tools, rlos itself and finally hydro data should be in same dir for running rlos.


Deprecated features

The following features are being marked as deprecated in this release.

It is no longer necessary to manually adjust RLOS grid dimensions in order to match the hydrocodes's ones, as it is now done automatically.

Pload no longer has paths inside it, uses dir argument instead for param passing from mother program rlos.

Proc and functions used for xrays are now commented out. 



The following documentation changes apply to this release:

This is the first release of RLOS to include release notes.





*****************************************************************************************************




How to run:

In order to run the program, IDL or GDL should be already installed. 

I. First of all, please put IN THE SAME DIRECTORY:

(1) RLOS itself (either XZ or YZ version, or both),

(2) parameter file rlos_params.txt,

(3) PLUTO's data output, in .dbl FORMAT (including all the big and little files of the hydrocode output, such as .out files, etc) and
 
(4) PLUTO's accompanying IDL routines (found in the PLUTO distribution, under '..../PLUTO/tools/IDL/' subdir). 

A simple way to do the above is to first run PLUTO, with .dbl data output, and then paste into the directory of the results the following: RLOS, its param file rlos_param.txt, and PLUTO's IDL routines.


II. Then we open RLOS and EDIT THE PATH NEAR THE BEGINNING, called datapath. We set the name of the path to: the directory of PLUTO data and RLOS. The first line of the following code portion should be edited, within RLOS:

;*************************************** 

DATAPATH='Q:\gitstuff\tempy210818SMALL';REPLACE WITH YOUR PATH!

cd, datapath

GDL_DIR='C:\Program Files (x86)\gnudatalanguage\gdlde'

PATH=!PATH+'C:\Program Files (x86)\gnudatalanguage\gdlde\'

!PATH=!PATH+datapath

pload, 1, dir=datapath

;***************************************


Therefore: Please set, within RLOS, datapath='....', where .... is the filesystem location of (RLOS, PLUTO's data output, PLUTO's IDL routines and rlos_params.txt).


III. Then please execute the first few lines of RLOS, in order to call and execute pload.pro, which comes with PLUTO, at least once. This is a requirement of pload, and loads some hydrocode data. Those lines should look similar to the following: (datapath should be changed to your own location of PLUTO data, etc)

;*************************************** 

DATAPATH='Q:\gitstuff\tempy210818SMALL'

cd, datapath

GDL_DIR='C:\Program Files (x86)\gnudatalanguage\gdlde'

PATH=!PATH+'C:\Program Files (x86)\gnudatalanguage\gdlde\'

!PATH=!PATH+datapath

pload, 1, dir=datapath

;***************************************

IV. Please edit the parameter file rlos_params.txt in order to setup RLOS for the run. 

In particular, the first entry (second line) shouuld be the same as datapath above. 

The rest of the params in rlos_params.txt should be edited according to the desired problem setup. Their values may also depend on the setup of the hydrocode run being imaged. 



V. Please compile RLOS and then run it. The synthetic image should emerge at the end of the execution. If debug output is desired, then please enable the relevant switch in the parameter file (debug_comments: set it, in the line that follows it, to 1).

*******************************************************************************************

rlos_params.txt

The contents of the parameter file are briefly explained here. Example values are given, for a sample run. Please adjust according to your model run. For more physical details, please see the relevant scientific paper of RLOS. Note that only odd lines (2, 4, 6, ...), from this parameter file, are used by RLOS. 

 
datapath (filesystem location of both hydro data and rlos) Entry IS INERT AT THE MOMENT, yet please do set it up)   

'Q:\gitstuff\tempy210818SMALL'

conditional_stop (0=NO, 1=YES: use stops along the execution line) 

0

debug_comments (0=NO, 1=YES: show debug interim results during execution)

0

sfactor_external (pload's shrink factor: reduces imaging grid resolution, please see pload.pro for more on this)

1.0

speedtweakfactor_external (ts speadtweak factor, globally multiplies matter speed, please see paper)

1.0

clight_override_external (0=NO natural ray speed clight value in cellls per hydro sec, 1=YES: override clight using clight preset value) 

0

clight_preset_external (clight override value, used only when override is active above)

0.1

jet_norm_velocity_external (jet nominal speed, as set in PLUTO.)

0.8

shotmin_external (MINIMUM snapshot to be loaded to RLOS)

2

shotmax_external (MAXIMUM snapshot to be loaded to RLOS)

22

phi2_external (ANGLE 2: elevation)

0.05D

phi1_external (ANGLE 1: azimuth)

1.57D

freqshiftchoice_external (FS SWITCH, 0 is off, 1 is on. See paper for details.)

0

dopplerchoice_external (DB switch, 0 is off, 1 is on. See paper for details.)

1.0

alphaindex_external (The spectral index of the presumed spectrum for the imaged jet system.)

2.0

nobs_external (Observing frequency. This is not used for now, since only density causes emission)

8000000000.0

NLONG_external  (max grid length. Please set to a higher value than the grid largest length size.)

150.0

plutolength_external (hydro model length unit in CGS cm)

10000000000

plutospeed_external (hydrocode speed unit in CGS cm/s)

30000000000

plutodensity_external (hydrocode density unit in CGS g/cm3)

0.00000000000000000000000167

plutocelllength_external (hydrocode cell length in CGS cm)

10000000000

*******************************************************************************************

Troubleshooting.

1. Please do not try to load more snapshots than available in the data. In rlos_params.txt, shotmin and shotmax correspond to the first and last snapshot respectively, as loaded by RLOS to RAM, and THEY SHOULD BE THERE in the hydro-data!

2. phi1 and phi2 angles are meant to vary from 0 to 90 degrees, or from zero to about 1.57 rads. In RLOS they are expressed in RADIANS, so angles vary from 0 to 1.57... rads. 

3. The combination of parameters in rlos_params.txt should correspond to a physically realistic setup. For example, they should NOT produce velocities higher than c, such as when ts=10 and ujet(injected)=0.25c. Please see the relevant paper! 

3. The considerable power of IDL/GDL can be used to improve the presentation and even production of results, but care should be exercised when editing RLOS, in order to not disturb the correct execution of the program. 

