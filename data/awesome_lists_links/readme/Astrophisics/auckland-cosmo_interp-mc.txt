# interp-mc
Archival patch to CosmoMC for "interp-mc" paper https://arxiv.org/abs/1012.5299 (published in JCAP), by Bouland, Easther and Rosenfeld.  

This code is provided purely for archival completeness and to set a good example by striving to avoid breaking the literature, 
as it is simply a patch to a now obsolete version of cosmomc.

If you were planning to use it, you would execute the following commands from the shell, in the CosmoMC source directory: 

> patch --dry-run -p1 -i code.patch [Checks patch will be applied without generating errors] 
> patch -p1 -i code.patch [Applies patch] 

Then run “make”.

Edit the params.ini file you are running with to include the following additional lines (possibly with other numerical values of your choice): 

do_interp = T 
interp_order = 4 
factor = 3 
cut = 8 
faction_cut = 0.2 

Then you could run CosmoMC as usual. 

Note that the repo contains a single .patch file.
