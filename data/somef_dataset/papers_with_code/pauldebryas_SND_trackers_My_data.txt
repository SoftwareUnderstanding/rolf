# EM shower reconstruction with the SciFi at SND@LHC
ML code for energy reconstruction of SND@LHC

## Instructions by Paul 

[in progress]

The files needed to do the energy reconstruction are:

-ship_tt.yml : Before anything, you need to create the needed environement with the folowing comand "conda env create -f ship_tt.yml" (to do once). Then, each time you log in, you need to activate the environment with the command "conda activate ship_tt".

-parameters.json: File in which the geometry of the SND is encoded. Numbers following "4X0" idicate the position of the TT planes used in the simulation. Initially, you need to put "Y_HALF_SIZE":21.5 and "X_HALF_SIZE":26.0.
The other geometry configuration (9X0, 5X0, ...) were used for previous analysis and will not be used here.

-utils.py: useful function, mainly to preprocess the raw MC data

-net.py: Class and function for the CNN. Class SNDNet describe the geometry of the network, you can play with it.

-coord_conv.py: useful functions for the implementation of the CoordConvmethod describe in the folowing paper: https://arxiv.org/pdf/1807.03247.pdf

-Preprocessing_script.py
-Paul_run_script.py

The other files (.swp, Debug.py, parameters.py) aren't necesarry.


## Useful links
 - FairShip: https://github.com/ShipSoft/FairShip
 - Intro on python, bash and git: https://hsf-training.github.io/analysis-essentials/
 - Stanford course on neural networks: http://cs231n.stanford.edu/2019/
 - Paul's thesis: https://lphe.epfl.ch/publications/theses/Master_Thesis_Paul_De_Bryas.pdf
 - SND@LHC proposal: https://cds.cern.ch/record/2709550/files/2002.08722.pdf

## Associated projects
 - SND@LHC, later (if approved) SHiP: event reconstruction in real time using the SCiFi as a sampling calorimeter. The SND is a detector made of emulsion bricks interleaved with SciFi planes, followed by a muon detector. It will be placed in the TI18 cavern, pointing at the ATLAS interaction region. It will detect neutrals (neutrinos + long-lived) produced at IP1. The same detector, in a bigger version, will constitute SHiP's SND (Scattering and Neutrino Detector). Emulsions are great at reconstructing neutrino interactions, with the "small" problem of being an integrating detector... That's why they are interleaved with SciFi planes, that will allow disentangling pile-up in the bricks as well as add time stamps. However, there's a certain level of analysis that can be done in real time using only SciFi information. The first part, EM shower reconstruction, has already been started, achieving energy reconstruction with a resolution of 5% at 100 GeV using a NN trained with electron events. But there's much more in the todo list: for example, gaining the ability of reconstructing lone tracks accompanying the EM shower. This helps in disentangling inelastic scatterings from v+e->v+e or DM+e->DM+e events.
 - [same introduction as above]: "optimising" / finalising the shower reconstruction algorithm: we have totally ignored the ghost hits problem, for example (the scifi planes will measure x/y with alternating layers), and we haven't studied what resolution on the shower energy and direction is needed to distinguish neutrino elastic scattering from dark matter scattering based on the different event kinematics due to nonzero dark matter mass, for example.
