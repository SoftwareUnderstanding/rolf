# RESIDUAL FEATURE EXTRACTION PIPELINE
A pipeline that carries out feature extraction of residual substructure within the residual images produced by popular galaxy structural-fitting routines such as GALFIT, GIM2D, etc. This pipeline extracts faint low surface brightness features by isolating flux-wise and area-wise significant contiguous pixels regions by rigourous masking routine. This routine accepts the image cubes (original image, model image, residual image) and generates several data products:

1. An Image with Extracted features.
2. Source extraction based segmentation map.
3. The background sky mask and the residual extraction mask.
4. A montecarlo approach based area threshold above which the extracted features are identified.
5. A catalog entry indicating the surface brightness and its error.

**Author:** Kameswara Bharadwaj Mantha

**email:** km4n6@mail.umkc.edu

**Publication:** 
Studying the Physical Properties of Tidal Features I. Extracting Morphological Substructure in CANDELS Observations and VELA Simulations.

**Corresponding Author:** 
Kameswara Bharadwaj Mantha

**Co-authors:** 
Daniel H. McIntosh, Cody P. Ciaschi, Rubyet Evan, Henry C. Ferguson, Logan B. Fries, Yicheng Guo, Luther D. Landry, Elizabeth J. McGrath, Raymond C. Simons, Gregory F. Snyder, Scott E. Thompson, Eric F. Bell, Daniel Ceverino, Nimish P. Hathi, Anton M. Koekemoer, Camilla Pacifici, Joel R. Primack, Marc Rafelski, Vicente Rodriguez-Gomez.

# INSTALLATION

Simply clone the github repository. The pipeline is present in the `Tidal_Feature_CAN_*` folder.

Requirements:
1. Please install `sep` from [sep documentation](https://sep.readthedocs.io/en/v1.0.x/)
2. I recommend running the pipeline in an `astroconda` environment. Please see [astroconda documentation](https://astroconda.readthedocs.io/en/latest/) for more details on how to install that environment. This should install all latest packages used by this pipeline.
3. In case you don't want to do step 2, here are the modules that you need: `optparse matplotlib astropy skimage warnings os`


# PIPELINE USAGE

You will use the python script: `Tidal_feature_finder.py`. This python file uses two other python scripts which have all the necessary functions that carry out the task.

**Usage:** Usage in your terminal is `python Tidal_feature_finder.py -p PARAMFILE.param`. Note that you have to provide the full path to PARAMFILE.param. Also, this information needs to be enclosed in single quotations.

**Example:** `python Tidal_feature_finder.py -p '~/path_to_parameter_file/test.param'`

#### IMPORTANT DETAILS about the PARAMETER FILE:
1. For ease of running the residual feature finding process, I have provided important levers used during the feature detection in a parameter file. NOTE this is NOT THE GALFIT PARAMETER FILE.
2. One has to write this parameter file for the galaxy one wishes to find the features.
3. The parameter file has several key words, each key word representing (each) lever that you can change. Please DONOT alter the keywords.
4. Each keyword and its value is arranged as "key = value". Please try to stick to this format. "=" sign is a must. In principle, spaces should be taken care of internally, feel free to test this and let me know if the code breaks.
5. The order of the keywords don't matter. The GALFIT output file and redshift of the galaxy are required and the rest of them are optional. By optional, I mean that the script uses default values (see below for additional description).

*Descriptions and cautions for the key words:*

(a) If any of the following keywords (except for the required fields) are provided as "none", then they would revert to their default values.

b) When mentioning the paths, please DO NOT provide "/" for the last folder. You don't need to create folders, just provide the paths and the code should create folders as needed. See the example parameter file provided.


#### KEYWORDS IN PARAM FILE

**I. gfit_outfile:** Please enter the path+filename to the galfit output file over which you want to find the residual features. This should be a fits file.

*Default value:* No defaults for this. This is a required field

**II. exp_stamp:** The cutout of the exposure map corresponding to the galaxy in question, where each pixel value holds the exposure time in seconds.

*Default value:* 5000 seconds

**III. sigma:** What significance above the sky background do you want to detect the features. If "sigma = 1", then the pixels whose values are above 1*sky are chosen towards computing the features.

*Default value: 2*

**IV. redshift:** Enter the redshift of the galaxy from the catalog. This is used to figure out the corresponding physical distance around the galaxy of interest.

*Default value:* None, this is a required field.

**V. boxcar_smooth_size:** The size of the boxcar2d filter used to smooth the residual image.

*Default value:* 3

**VI. forced_segvalues_galfitted:** In case you are wanting to extract residual features for multiple galaxies in the image. First, make sure that these galaxies are GALFITTED. Then go to the source extract folder and spot the feature extraction image. Enter the segmap values corresponding to the sources you want to force the extraction of features in a consistent fashion to primary galaxy of interest. For example forced_segvalues_galfitted = 9, will perform the feature extraction on the source number 9 by repeating the exact same process performed on the primary galaxy (at the center of the image).

*Default values:* None

**VII. forced_segvalues_not_galfitted:** In some cases, extended features are identified as separate sources in the image. They might get masked if not taken care of and will be omitted during feature extraction. Therefore, if one wishes to forcibly unlock regions that are not GALFITTED, then please provide their corresponding segmentation values. For example forced_segvalues_not_galfitted = 5 will unlock the segmentation region corresponding to object 5.

*Default values:* None.

**VIII. inner_mask_semi_maj:** The semi major axis multiplier as the width of the ellipse (diameter) that masks the central region.

*Default value:* 1.5 * object's semi-major axis

**IX. inner_mask_semi_min:** The semi minor axis multiplier as the height of the ellipse (diameter) that masks the central region.

*Default value:* 1.5 * object's semi-minor axis

**X. outer_phy_mask_size:** the size of the outer circular aperture [in kpc] after which the feature extraction is not performed.

*Default value:* 30 (means 30 kpc).


**XI. run_MC:** In order to choose features that are significant above a random noise expectation, the machine will perform a Monte Carlo simulation of extracting the features that show up if you just have a sky background. If you are running the tidal feature finder for the first time on a galaxy, have this toggled to True. Once it runs, it generates necessary files and stores them for future purpose. It computes an Area threshold and stores it in "MC_path" folder under "A_thresh" folder. You will notice that a text file with the galaxy id (you provided) is created. If you open it, there will be one number which is the area threshold above which a region is statistically unlikely to be caused by sky background. Also, in the same folder, it creates some plots with the random noise image and how it looks like if we applied our tidal feature finder on just noise. This is a diagnostic plot to make sure we are not doing any thing crazy wrong.

*Default value:* 'True'


**XII. plot_destin:** This path will store the key figure, where the residual feature is overlaid on the host galaxy. Please provide the path here.

*Default value:* current working directory, where a new folder is created

**XIII. SB_info_path:** At the end of the residual feature finding, the script computes the surface brightness of the features. Please provide a path in which csv files with the surface brightness information can be stored. The csv file will be structured as follows: ID, Surface brightness, Surface brightness error

*Default value:* current working directory, where a new folder is created

**XIV. sep_sigma:** This is the significance above which you want the sources in your image to be detected. Note that this is used exclusive for source detection.

*Default value:* 0.75

**XV. sep_min_area:** The minimum area of significant pixels to be called as a source.

*Default value:* 7

**XVI. sep_dblend_nthresh:** Number of threshold levels used for de-blending sources.

*Default value:* 32

**XVII. sep_dblend_cont:** The deblend minimum contrast level used for deblending of sources. Please see SExtractor definitions or go to SEP python webpage.

*Default value:* 0.001

**XVIII. sep_filter_kwarg:** The key word indicating what filter do you want to use during the source extraction process. The available key words are "tophat", "gauss", "boxcar".

*Default value:* tophat

**XIX. sep_filter_size:** What is the appropriate filter size you want to use during the source extraction process.
For tophat --> it is the radius
For gauss --> it is the fwhm (in pixels)
For boxcar --> It is the box size.

*Default value:* for all filters, it is 5.

**XX. sep_make_plots:** If you want to see the source extraction output, please mention True here.

*Default value:* True

**XXI. sep_plot_destin:** The destination folder to store the plots, if you decide to the source extraction output.

*Default value:* current working directory, where a new folder is created

**XXII. fits_save_loc:** save location where the fits image files generated during the feature extraction process are stored.

*Default value:* current working directory, where a new folder is created

#### Work in progress
Note that you will notice a couple of more keywords in the example parameter file.
These correspond to a Voronoi Tesselation of the residual features that will incorporated in the upcoming versions of the feature extraction.

Feel free to delete these key words from the parameter files you create. The default values for making Voronoi Tesselation
are set to False, so it shouldn't cause any issue.
