When using this algorithm for publications, please acknowledge the work presented in:
http://arxiv.org/abs/17soon

========================================================
INSTALLATION: 
- Download and unzip the SLIT.zip file
- In command line, open the SLIT folder with "cd SLIT"
- Install the python package:
"python setup.py install"

========================================================
TESTS:
In SLIT, open the "Examples" folder and execute either of the examples:
"python Test_SLIT.py" to test the reconstruction of a deblended lensed source.
"python Test_SLIT_HR.py" to test the reconstruction of a deblended lensed source at higher resolution.
"python Test_SLIT_MCA.py" to test the joint separation of lens and source light profiles and reconstruction of the source.
"python Test_sparsity.py" to reproduce the non-linear approximation error plot from the paper to show the sparsity of lensed and unlensed galaxies.

Inputs provided in the "Files" folder:
"source.fits": the source used to generate simulations in Test_SLIT.py and Test_SLIT_MCA.py.
"Source_HR.fits": the source used to generate simulations in Test_SLIT_HR.py.
"Galaxy.fits": the lens galaxy light profile used in Test_SLIT_MCA.py.
"kappa.fits": the lens mass density profile used in all simulations.
"PSF.fits": the TinyTim PSF used in all simulations.
"Noise_levels_SLIT_HR.fits": noise levels used in "Test_SLIT_HR" saved as a fits cube. They are usually computed in the algorithm, but we provide them as inputs to save time.
"Noise_levels_SLIT_MCA.fits": noise levels used in "Test_SLIT_MCA" saved as a fits cube. They are usually computed in the algorithm, but we provide them as inputs to save time.
"Noise_levels_SLIT.fits": noise levels used in "Test_SLIT_HR" saved as a fits cube. They are usually computed in the algorithm, but we provide them as inputs to save time.

Results are displayed in pyplot windows.

Users may change the input parameters via the same files and are encouraged to use them for further applications.

========================================================
HOW TO USE:
We provide a Test_for_user.py file, that users may fill in with their own inputs and where the commands are a bit more detailed than in the other Test files.

SLIT needs very little amount of inputs. It requires the input image along with lens mass profile and a PSF. The user then has to chose a maximum number of iterations after which the algorithm will stop if not converged and a image size ratio to the input image to set the resolution of the reconstructed source.

Please do not hesitate emailing me for support via github, or at: remy.joseph@epfl.ch.




Contact GitHub API Training Shop Blog About 

© 2017 GitHub, Inc. Terms Privacy Security Status Help 

