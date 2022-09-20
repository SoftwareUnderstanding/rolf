# stepped_luneburg
Python 2.7.X module for modeling of a stepped Luneburg lens for all-sky surveys

## Code Citation

The code is registered at the [ASCL](http://ascl.net "ASCL") with the identifier [ascl:1809.014](http://ascl.net/1809.014) and should be cited in papers as:

    Carney, M.T., Kenworthy, M.A., 2018, Astrophysics Source Code Library, record ascl:1809.014

Define the Module Path
----------------------

- Option 1: Manually include the path in a personal script
    - when importing modules at the top of your Python script, add the lines: </br> 
    `import sys` </br>
    `sys.path.append("/my/path/to/luneburg_lens/model")`

- Option 2: Globally include the path by adding it to the system variable $PYTHONPATH
    - bash - add the following line to the .bashrc file: </br>
    `export PYTHONPATH="/my/path/to/stepped_luneburg/model:$PYTHONPATH"`
    - c-shell - add the following line to the .cshrc file: </br>
    `setenv PYTHONPATH ${PYTHONPATH}:/my/path/to/stepped_luneburg/model`


Known Issues
----------------------

- Setting stepped_luneburg() parameter 'center' to anything but default ([0,0,0]) causes plotting error
    - **To do**: debug 'center' parameter

- The plot = True option in stepped_luneburg() becomes a memory hog if the number of rays approaches 1000 or greater because all ray trajectories and ray-interface intesections are stored for plotting
    - **Solution:** set plot = False for large nrays


Directory Structure
----------------------

Contents:  

    stepped_luneburg
    |
    |-- model
    |   |-- luneburg_lens.py        : Contains stepped_luneburg() function for running Luneburg lens model
    |   |-- lens_setup.py           : Functions to initialize the Luneburg lens
    |   |-- lens_ray_tracing.py     : Functions called during the ray tracing routine
    |   |-- lens_plotting.py        : Functions to plot the Luneburg lens
    |   |-- enclosed_intensity.py   : Script with function enc_int() to create a plot of enclosed intensity vs angle based on the output of stepped_luneburg()
    |   |-- intensity_map.py        : Script with function int_map() to create an intensity map and magnitude histogram based on the output of stepped_luneburg()
    |   |-- orion.dat               : Example star pattern that can be used for ray tracing
    |   |-- __init__.py
    |
    |-- scripts
    |   |-- luneburg_wrapper.py                 : wrapper script for stepped_luneburg() function     
    |   |-- enc_int_wrapper.py                  : wrapper script for enc_int() function
    |   |-- int_map_wrapper.py                  : wrapper script for int_map() function
    |   |-- plot_enc_int_vs_theta_multi.py      : script to plot the enclosed (cumulative) intensity as a function of the angle away from the focal point
    |   |-- plot_theta_vs_exp_multi.py          : script to plot the the angle away from the focal point as a function of the Luneburg power-law exponent
    |   |-- plot_int_vs_exp_multi.py            : script to plot the total output intensity as a function of the Luneburg power-law exponent
    |
    |-- plots
    |   |-- fig1                    : directory to reproduce Figure 1 of Luneburg paper
    |   |-- fig2                    : directory to reproduce Figure 2 of Luneburg paper
    |   |-- fig3                    : directory to reproduce Figure 3 of Luneburg paper
    |   |-- fig4                    : directory to reproduce Figure 4 of Luneburg paper
    |   |-- fig5                    : directory to reproduce Figure 5 of Luneburg paper
    |   |-- fig6                    : directory to reproduce Figure 6 of Luneburg paper
    |   |-- fig7                    : directory to reproduce Figure 7 of Luneburg paper
    |
    |-- README.md
    |-- LICENSE
    |-- .gitignore
    

# Description of Code Modules

luneburg_lens.py | stepped_luneburg()
-----------------------

Create a stepped Luneburg lens model: a spherical lens of varying index of refraction

    Default parameters:
    stepped_luneburg(outfile="luneburg_output.dat",steps=20.,exp=0.5,nrays=100.,amp_cut=0.01,center=[0.,0.,0.],mode=None,modify=False,plot=True,figoutfile="luneburg_lens_raytrace.pdf",verbose=False,logfile="luneburg_log.dat")
    
    This routine traces the path of a wavefront of rays through the Luneburg lens using Snell's Law at each lens layer. The radius of the lens surface is normalized to 1. Normalized radii for each lens layer are calculated based on the number of steps provided. The lens index of refraction is calculated as n = (2-r^2)^exp and increases in discrete steps from the lens surface to the lens center with the maximum refractive index at center. 
    
    The mode for the wavefront of rays is initialized such that all incoming rays are parallel (i.e. incoming from infinity) and each ray has an amplitude of 1. Rays enter the lens if they are incident on the top hemisphere. Rays incident on the bottom hemisphere of the lens are discarded. All rays that enter the lens are propagated until the ray exits the lens or falls below a designated amplitude threshold. 
    
    The position, direction, and intensity for rays that exit the bottom hemisphere of the lens are stored and accumulated so that an image can be simulated on the bottom surface of the lens. These ray parameters are written to an output file for later processing.
 
    A four-panel figure can be created to visualize ray tracing through the Luneburg lens. 
      Plot a) shows the lens layers and the position of the incident wavefront in the x-y plane (side view). 
      Plot b) shows the lens layers and the position of the incident wavefront in the x-z plane (top view). 
      Plot c) shows the lens layers and each ray propagating through each lens layer in the x-y plane (side view). Rays striking the outermost surface of the lens are shown with a large orange dot. Rays striking any lens layer interface are shown with a small red dot. Rays exiting the lens are shown with a yellow dot. Rays marked with a black 'X' have been dropped from the routine due to the amplitude cutoff or due to exiting the top hemisphere of the lens.
      Plot d) shows a 2D projection of the rays exiting the bottom hemisphere of the lens in order to simulate an image produced by the lens. Exiting rays are shown as small blue dots, and an alpha value is applied to each dot that corresponds to intensity (i.e. exiting rays with less intensity are more transparent).
        ***note: if mode='stars' then only the stars in the star pattern are plotted in Plots a), b), c), and an unplotted 'grid' pattern of rays is rotated to the stellar position and used to initialize the wavefront of incoming rays from each star

    Optionally, this routine can calculate the lens layer indices of refraction with a modified refractive index power law equation where n = (2-r^(2*exp)) for performance comparison to the original refractive index power law equation.
    
    Optionally, this routine can write all terminal output to a log file if verbose = True.

    ==========
    Parameters
    
    outfile: [string] name of the output file (default = "luneburg_output.dat")
    
    steps: [float] number of steps/layers for the Luneburg lens model (default = 20)
    
    exp: [float] power-law exponent for the varying index of refraction of the stepped lens model (default = 0.5)
            
    nrays: [float] numbers of rays to initialize for the wavefront incident on the Luneburg lens; should be a square number (default = 100.)

    amp_cut: [float] amplitude cutoff value for rays propagating through the lens: rays with amplitude below this value after Snell's law are dropped by the program (default = 0.01, 1% amplitude of ray prior to Snell's law)

    center: [list of floats] x, y, and z coordinates of the lens center (default = [0.,0.,0.])
        *** CAUTION: non-zero centers are not properly implemented yet ***

    mode: [string] initial pattern of the wavefront incident on the upper hemisphere of the Luneburg lens (default = None)
        options are 'stars', 'random', 'grid', or 'line'
        'stars' - star pattern on the sky (default = Orion) that can be manually updated/expanded by the user
        'random' - random distribution of points in the x-z plane within a given radius
        'grid' - uniform grid of points in the x-z plane within a given radius, nrays should be a square number
        'line' - single line of points along the x axis

    modify: [boolean True/False] option to modify power law used to calculate the refractive index: True for modified power law, False for original power law (default = False)
        original: n = (2-r^2)^exp
        modified: n = (2-r^(2*exp))

    plot: [boolean True/False] option to show plot of ray tracing through the lens and save the figure (default = True)
        *note: False recommended if nrays is very large
    
    figoutfile: [string] name of the output figure file if plot = True (default = "luneburg_raytrace.pdf")
    
    verbose: [boolean True/False] option to print out more information to the terminal and save to a log file (default = False)
    
    logfile: [string] name of log file containing terminal output if verbose=True (default="luneburg_log.dat")
    
    ========== 
    Usage
    
    # import the stepped_luneburg function
    >> from luneburg_lens import stepped_luneburg

    # run a Luneburg model for 'grid' wavefront pattern of incoming rays
    >> stepped_luneburg(mode='grid'):

    # run a Luneburg model for 'grid' wavefront pattern of incoming rays with a non-zero center 
    *** CAUTION: non-zero centers are not properly implemented yet ***
    >> stepped_luneburg(mode='grid',center=[0.1,0.1,0.1]):

    # run a Luneburg model for 'grid' wavefront pattern of incoming rays without producing a plot
    >> stepped_luneburg(mode='grid',plot=False):

    # run a Luneburg model for 'line' wavefront pattern of incoming rays for a lens with 40 layers, 1000 rays, and amplitude cutoff at 10% of the ray amplitude prior to Snell's law
    >> stepped_luneburg(mode='line',steps=40.,nrays=1000.,amp_cut=0.1):
        
    # run a Luneburg model for 'random' wavefront pattern of incoming rays with the modified refractive index power law equation and power law exponent 0.6 
    >> stepped_luneburg(mode='random',exp=0.6,modify=True):
   
    # run a Luneburg model for 'stars' wavefront pattern of incoming rays, save to myfile_out.dat, save figure to myplot.pdf, and save log file mylog.dat
    >> stepped_luneburg(outfile="myfile_out.dat",mode='stars',figoutfile="myplot.pdf",verbose=True,logfile="mylog.dat"):
 
    ==========    
    

enclosed_intensity.py | enc_int()
-----------------------

Calculate the cumulative enclosed intensity as a function of angle away from the Luneburg lens focal point. 

    Default parameters: 
    enc_int(infile=None,outfile="luneburg_enc_int.dat",frac=0.5, mode=None, star_num=None,plot=True,figoutfile="luneburg_enc_int.pdf"):
    
    This routine uses the output data file from luneburg_lens.py to calculate the enclosed intensity of rays exiting the bottom hemisphere of the lens as a function of the angle away from the expected wavefront focal point. The data is saved to an output file that records the angle away from the focal point, the total enclosed intensity at each angle, and the normalized enclosed intensity at each angle. The data can also be plotted with normalized cumulative intensity as a function of log of the angle from the focal point.
    
    ==========
    Parameters
    
    infile: [string] name of the input file for ray position and intensity data (default = None)

    outfile: [string] name of the output file for angle and enclosed intensity data (default = "luneburg_enc_int.dat")
    
    frac: [float] fraction of total enclosed intensity to mark for plotting; measure of lens performance (default = 0.5)
    
    mode: [string] initial pattern of the wavefront incident on the upper hemisphere of the Luneburg lens (default = None)
        options are 'stars', 'random', 'grid', or 'line'
        'stars' - star pattern on the sky (default = Orion) that can be manually updated/expanded by the user
        'random' - random distribution of points in the x-z plane within a given radius
        'grid' - uniform grid of points in the x-z plane within a given radius, nrays should be a square number
        'line' - single line of points along the x axis
            *note: if mode='stars' and star_num = ##: intensity output from the entire star pattern is read in from infile, so other stars in pattern will contaminate enclosed intensity function for star ## in star pattern

    star_num: [int] integer corresponding to the desired star from a user-provided star pattern; only required if mode='stars' (default = None)
    
    plot: [boolean True/False] option to show plot of normalized enclosed intensity as a function of log(angle) and save the figure (default = True)
    
    figoutfile: [string] name of the output figure file if plot = True (default = "luneburg_enc_int.pdf")

    ==========    
    Usage
     
    # import the enc_int function
    >> from enclosed_intensity import enc_int

    # calculate enclosed intensity from myfile.dat for 'grid' pattern, plot results
    >> enc_int(infile="myfile.dat",mode='grid'):
   
    # calculate enclosed intensity from myfile.dat for a specific star from the user-provided 'stars' pattern
    >> enc_int(infile="myfile.dat",mode='stars',star_num=2):
   
    # calculate enclosed intensity from myfile.dat for a 'line' pattern, write output to myfile_out.dat, and plot results to myplot.pdf
    >> enc_int(infile="myfile.dat",outfile="myfile_out.dat",mode='line',figoutfile='myplot.pdf'):
 
    # calculate enclosed intensity from myfile.dat for a 'grid' pattern with a high fraction of total intensity marked for plotting
    >> enc_int(infile="myfile.dat",mode='grid',frac=0.9):
 
    ==========    

    
intensity_map.py | int_map()
-----------------------

Create a 2D bullseye-style intensity map of haloes produced by the Luneburg lens.

    Default parameters:
    int_map(infile=None,outfile="luneburg_int_map.dat",pixels=500.,rings=100.,frac=0.5,sky_mag=13.,figoutfile="luneburg_int_map.pdf",mag_hist_plot=True,verbose=False)
    
    This routine uses the output data file from luneburg_lens.py to plot a 2D intensity map with a central region containing a user-supplied fraction of the total output intensity. The image is then normalized such that the central region is set equal to 1, and the remaining output intensity is recorded in concentric rings. The number of rings depends on the 'frac' parameter, as the bin size for the rings is set equal to the radius containing fraction 'frac' of the total output intensity. The final intensity map is plotted in log scale and tick labels are set to show the maximum radius of the pixel map normalized to 1.
    
    The aim of this routine is to show the strength of haloes produced by the Luneburg lens relative to the user-supplied fraction of the total output intensity. The user-supplied fraction 'frac' should reflect what the user considers to be sufficient lens performance for imaging. The intensity map is plotted in log scale for easy comparison to stellar magnitudes, with a color bar that displays the magnitude of the haloes as magnitude = -2.5*log(intensity). Haloes that have relative magnitudes greater than (i.e. dimmer) the sky brightness background magnitude are washed out, as they will not be detectable over the sky background noise transmitted through the lens.
    
    Example: The user-supplied fraction is 0.5. The user considers capturing 50% of the total output intensity from the Luneburg lens ray tracing routine to be sufficient lens performance for imaging. The central region of the intensity map will contain 50% of the total output intensity and the image is normalized so that the central region is equal to 1. The remainder of intensity is stored in concentric rings that correspond to lens haloes, each with a width equal to the radius of the central region containing 50% of the output intensity. In log scale, if any of the haloes have a relative magnitude greater than (i.e. dimmer) the sky brightness background magnitude (ring_mag > 'sky_mag') then they are set equal to 'sky_mag' as they will not be visible above the sky background. Haloes that are brighter than the sky background (ring_mag < 'sky_mag') will remain visible and potentially contaiminate the image, depending on their magnitude.
    
    Optionally, this routine can plot a magnitude vs. radius histogram with the mag_hist_plot option. The plot uses the data from the intensity map to show the halo positions and halo strengths relative to the normalized central region containing 'frac' of the total intensity output.
    
    ==========
    Parameters
    
    infile: [string] name of the input file for ray position and intensity data (default = None)

    outfile: [string] name of the output file for 2D map Luneburg lens intensity output in ascii format (default = "luneburg_int_map.dat")
    
    pixels: [float] number of pixels per edge for intensity map of Luneburg lens output (default = 500)
    
    frac: [float] fraction of total Luneburg lens output intensity to be normalized to one; a lens performance threshold (default = 0.5)
    
    sky_mag: [float] sky brightness background magnitude (default = 13)
    
    figoutfile: [string] name of the output figure file (default = "luneburg_int_map.pdf")
        
    mag_hist_plot: [boolean True/False] option to create a magnitude vs radius histogram plot from the intensity map, output file name is figoutfile"_hist".extension
    
    verbose: [boolean True/False] option to print out more information to the terminal (default = False)
    
    ==========
    Usage 
    
    # import the enc_int function
    >> from intensity_map import int_map

    # create intensity map from myfile.dat and plot magnitude vs. radius histogram
    >> int_map(infile="myfile.dat"):
   
    # create intensity map from myfile.dat and save to myplot.pdf, but no histogram plot 
    >> int_map(infile="myfile.dat",figoutfile="myplot.pdf",mag_hist_plot=False):
 
    # create intensity map from myfile.dat with 1000 pixels
    >> int_map(infile="myfile.dat",pixels=1000):

    # create intensity map from myfile.dat with 1000 pixels, with central region containing 80% of Luneburg lens output intensity
    >> int_map(infile="myfile.dat",pixels=1000,frac=0.8):
 
    ==========

