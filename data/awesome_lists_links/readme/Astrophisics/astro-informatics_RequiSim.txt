#      _RequiSim_                    #



# Background #

**statement**: I hope that you enjoy _RequiSim_, and find it useful.
Please contact Peter Taylor (**peterllewlyntaylor at gmail dot com**) 
if there are problems, or if you need the data files for 
surveys other than Euclid.
If you use _RequiSim_ please remember to cite the
two papers listed below.


**name**: _RequiSim_

**version**: 1

**purpose**: Computes the Variance Weighted Overlap which is
a measure of the bias on the lensing signal 
from power spectrum modelling bias.

**attribution**: Peter Taylor, Mullard Space Science Laboratory,
University College London, 2018

**cite**: 

- __Preparing for the Cosmic Shear Data Flood: Optimal Data Extraction and
Simulation Requirements for Stage IV Dark Energy Experiments__: The formalism
      in this code is developed.
- __Testing the Cosmic Shear Spatially-Flat Universe Approximation with GLaSS__: _GLaSS_, which produced the data files, is described.



**assumptions**

- Bias on the power spectrum is Gaussian
    with a covariance descirbed by a knowledge matrix
    that does not change at different points in
    cosmological parameter space.
- Euclid wide-field survey. We can provide
    data files for other surveys on request.
    Please email **peterllewlyntaylor at gmail dot com** 



**explanation** 
    
   _RequiSim_ is used to compute requirements on power spectrum simulations for
    upcoming cosmic shear experiments. The user must provide a knowledge matrix
    which describes the covariance in the bias on the power spectrum.
    See Taylor et al. in prep for more details.


**run**
    
   You can import the functions defined in `RequiSim.py` as needed. A demo script 
    showing how to use these is provided. It is:

    sample_run_script.py



**python dependencies**

- numpy 
- matplotlib
- scipy
- math
- random



# Functions #

All functions are internal to the program except the four listed
below, defined in `RequiSim.py`, which
accept external input and give external output.
    
**`P_VWPO`** 

- **purpose**

    Compute the lensing signal bias due to power spectrum modelling error

- **inputs** 
            
    **`knowledge_matrix`**: The covariance between power spectrum bins.
            Format described below.

    **`l_cut`**: The l-mode angular scale cut. Default = 3000.
            
    **`frac_captured_info`**: Fraction of the variance to capture. Used
            for dimensional reduction so we don't have to work in a 225 
            dimensional parameter space. Default is 99%. This is well tested
            and code is fast, so there should be no reason to change.

    **`n_samples`**: The variance weighted overlap is a marginalised quantity
            computed from drawing samples from a distribution function. This variable 
            describes how many samples to draw. Default = 5000. Precision of variance 
            weighted overlap is ~1% at this default.


- **outputs**

    The variance weighted overlap.

**`plot_k`**


- **purpose**: 

   	Gives a visual representation of the knowledge matrix.

- **inputs**:

    **`knowledge_matrix`**: Input format described below.

- **ouputs**:
       
       saves a plot called knowledge_matrix.png which shows the bias on different
            cells on the power spectrum P(k,z) in k-z space.



**`get_k_cell_boundaries`**

- **purpose** 
    
    Show the boundaries of the power spectrum cells in k. Needed
            if you want to provide a custom knowledge matrix.

- **inputs** 
   
   	None

- **outputs**

       Numpy array showing boundaries in k [h Mpc^{-1}]




**`get_z_cell_boundaries`**

- **purpose** 

    Show the boundaries of the power spectrum cells in z. Needed
            if you want to provide a custom knowledge matrix.

- **inputs** 

    None

- **outputs** 

    Numpy array showing boundaries in z 







# Knowledge Matrix Input Format #

The knowledge matrix must be given as a 2D numpy array. This can be loaded with
    `np.loadtxt()` function or read from an interactive Python session or script. The 
    dimensions should be (225,225) since there are 15 grid cells in both k and z. The
    diagonal elements give the bias on the cells and the off diagonal gives the correlation
    in the bias between cells. Cells are order by the following relation:

    CELL_NUMBER = 15 * Z_CELL_NUMBER + K_CELL_NUMBER

The cell boundaries for each cell can be displayed by running `get_k_cell_boundaries()`
    and `get_z_cell_boundaries()` in _RequiSim_.





   
   
   
   
   
