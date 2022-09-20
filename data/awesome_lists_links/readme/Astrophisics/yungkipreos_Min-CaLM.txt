# Min-CaLM
The Min-CaLM python package can perform automated mineral compositional analysis on debris disk spectra. It will determine the minerals that are present and what their relative abundances are within the debris disk. This code is used in a paper that has been submitted to the American Journal of Undergraduate Research (AJUR), titled "An Unbiased Mineral Compositional Analysis Technique for Debris Disks" by Yung Kipreos and Dr. Inseok Song. Min-CaLM stands for "Mineral Compositional Analysis using Least Square Minimization".

## Getting Started
To use Min-CaLM's code, the following python libraries must be imported: numpy, astropy.io, pylab, scipy.optimize, matplotlib, scipy, decimal.

### Debris Disk Spectrum Requirements
Min-CaLM has several assumptions regarding the format of the target debris disk spectrum. It assumes that the spectrum's data file is in a two-column format where the first column is the wavelength data and the second column is the flux data, and also that both of the columns are in increasing order (according to the wavelengths data column). The wavelength range of the debris disk spectrum should be around from ~5 to ~45 microns.

To perform mineral compositional analysis on a debris disk spectrum, there must be prominent silicate mineral features present. Below is an example of a debris disk spectrum with no silicate mineral features:

<img src="/HD192758_Debris_Disk_Spectrum.png" width = 300 >

And below is an example of a debris disk spectrum that displays prominent silicate mineral features:

<img src="/HD15407_Debris_Disk_Spectrum.png" width = 300 >

### Running the Min-CaLM Program
**It is reccommended to run the Min-CaLM program through the command line. Download both the Min-CaLM.py file and the mineral spectra files into the same directory.** There are some example debris disk spectra that can be downloaded as well. The Min-CaLM.py file, mineral spectra files, and example debris disk spectra files can be all found in this Git-hub repository. 

To run Min-CaLM, go to the command line and navigate to the directory that contains all of the above files. Then type "ipython" into the command line. Now that you are in ipython, type "run Min-CaLM.py" to run the program. The program will ask you to "Please input the target's spectrum (target spectrum must be kept in the same folder as Min-CaLM.py):". Type in the file name of the debris disk spectrum into the command line and press enter. The program will then display a figure of the recreated spectrum (blue) plotted over the original debris disk spectrum (red). In the command line, a table will be displayed that shows the minerals that were determined to be present in the disk and their relative abundances.

# Tutorial 1
This tutorial is a more indepth description of how to use the Min-CaLM program to perform mineral compositional analysis on debris disk spectra. This tutorial assumes that the star's photosphere blackbody contributions has aleady been removed from the spectrum. The resulting spectrum is the debris disk spectrum that contains the mineral spectra and the blackbody spectrum produced by the heated dust/debris. Here, I will use the debris disk around BD+20 307 as an example. BD+20 307 is a dusty binary star system in the Aries constellation. 

The BD+20 307 debris disk data file is in a two column format, a section of which is shown below. The left column is the wavelength data and the right column is the flux data. Notice that the data is in ascending order according to the wavelength data. 

<img src="/BD+20_307_Sample_Data.png" width = 300 >

And this debris disk spectrum looks like:

<img src="/BD+20 307_Spectrum.png" width = 300 >

Before the Min-CaLM program can be used on the debris disk data, the dust's blackbody spectrum must be removed. To remove the blackbody spectrum, type in the estimated temperature of the dust into line . This line of code is shown below, with a "#ENTER DUST BB TEMP HERE" comment next to it. Here the dust temperature is set to 502 K. To change the set temperature, just replace the "502" with a different temperature (in Kelvin).

<img src="/BB_Removal_Code.png" width = 400 >

*Note: If the dust blackbody spectrum has already been removed from the debris disk spectrum, then the section titled "DUST BLACKBODY REMOVAL" must be commented out along with the first while loop in the "MULTIPLYING THE OVERALL SPECTRA BY THE BB CONTINUUM" section. 

Now the Min-CaLM program is ready to be used. In command line, navigate to the directory in where Min-CaLM.py, the mineral spectra files, and the debris disk spectrum are located. In this example they are located in a folder called "Min-CaLM_Files".

<img src="/1.png" width = 300 >

Type "ipython" into the command line, and afterwards type "run Min-CaLM.py" to run the program. 

<img src="/3.png" width = 300 >

After around ~20 seconds, a prompt will appear that reads "Please input the target's spectrum (target spectrum must be kept in the same folder as Min-CaLM.py):". 

<img src="/4.png" width = 300 >

After this prompt appears on the screen, type in the debris disk file name into the command line and hit enter. Here the file name is "BD+20_307_Debris_Disk.txt".

<img src="/5.png" width = 300 >

A figure will appear with the recreated debris disk spectrum by Min-CaLM (in blue) plotted over the original debris disk spectrum (in red). 

<img src="/BD+20 307 (BB = 502).png" width = 300 >

The recreated spectrum is calculated by multiplying each mineral spectrum by its relative abundance, and then adding each of the weighted spectra together. The relative abundance of each mineral within the disk, as calculated by Min-CaLM, is displayed in a table in the Terminal window.

<img src="/6.png" width = 300 >














