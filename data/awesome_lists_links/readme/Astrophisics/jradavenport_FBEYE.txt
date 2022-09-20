FBEYE: an IDL suite for analyzing Kepler light curves and validating flares
=====

FBEYE is the "Flares By-Eye" detection suite, developed by the 
Low-mass stellar astronomy research group at the University of 
Washington. It incorporates algorithms from many authors, and 
the work from multiple PhD thesis projects under the supervision 
of Prof. Suzanne L. Hawley. 


**Note 1:** FBEYE requires the following IDL libraries to function:

- [jradavenport library](https://github.com/jradavenport/jradavenport_idl)
- [Coyote library](https://github.com/idl-coyote/coyote)
- [IDLAstro library](https://github.com/wlandsman/IDLAstro)

**Note 2:** FBEYE has been tested to work on IDL v7.0 - v8.4



FBEYE will work on any 3 column light curve that contains time,flux,error. However, the success of flare identification is highly dependent on the [smoothing routine](../jradavenport_idl/softserve.pro), which may not be suitable at present for all sources. Therefore, if FBEYE returns too many erroneous flare candidates your light curve may need to be preprocessed.

The repository also contains the [analytic flare model](aflare.pro) described in the paper. While FBEYE does not call this code directly, fitting this model to the flare candidate events can be useful for 1) validating the events, and 2) determining complex event structure. Future versions of FBEYE *may* feature this functionality built in.


(C) 2011, 2012, 2013, 2014, 2015 James R. A. Davenport </br>

If you use this code please cite our paper: [Davenport et al. 2014](http://arxiv.org/abs/1411.3723). Send questions or comments to @jradavenport