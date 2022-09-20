# MultiModes
- Author: David Pamos Ortega - PhD Student - University of Granada (UGR) -
- Thesis Directors: Dr. Juan Carlos Suárez Yanes - University of Granada (UGR) -  and Dr. Antonio García Hernández - University of Granada (UGR) -
- Expert contributor: Dr. Javier Pascual Granado - Institute of Astrophysics of Andalusia (IAA) -
- If using MultiModes, please cite the author this way: Pamos Ortega, D. et al. 2022 (https://doi.org/10.1093/mnras/stac864)

# What is it?
MultiModes is a python code to extract the most significant frequencies of a sample of classical pulsating stars

# Input
- Directory with light curves in format .fits, corrected from 'outliers' and 'nan' values. Headers: 'TIME' and 'PDCSAP_FLUX'
- ini.txt with the initial parameters: 
  - sim_fit_n: Number of simultaneous peaks to be fit before extracting to the original light curve for obtaining the residual: 20 by default
  - max_freq: Maximum value of the analysed frequencies domain: 100 c/d by default (delta Scuti stars)
  - os_ratio: oversampling factor: 5 by default
  - stop: Stop criterion, FAP or SNR: SNR by default
  - min_snr: Minimum signal to noise ratio: 4 by default (Breger 1993)
  - max_fap: Maximum value of the False Alarm Probability: 0.01 by default (Balona et al. 2014)
  - tail_per: Minimum  frequency of the tail of the periodogram: 80 c/d by default (delta Scuti stars)
  
# Output
- Directory 'results', containing subdirectories corresponding to every analysed light curve. Each subdirectory contains:
  - file best_modes.dat, containing the values of the most significant frequencies, amplitudes, phases, corresponding errors and FAPs/SNRs
  - file lc.dat, the light curve in format .dat for using with other codes, such as SigSpec (Reegen 2007)
  - file pg.dat, the periodogram of the original light curve
  - LC.png, the plot of the light curve
  - LS.png, the Lomb-Scargle periodogram
  - LS_n.png, the periodogram obtained after a number n of extracted peaks
  - res.dat, with the final residual after extracting all the most significant frequencies

# Pre-Installed Packages Requirements
- python 3.8.5
- numpy 1.19.2
- matplotlib 3.3.2
- pandas 1.1.2
- astropy 4.0.2
- lmfit 1.0.2
- scipy 1.5.2

# What does it do
MultiModes takes as input a directory with light curves, corrected from 'outliers' and 'nan' values, in fits format, with headers 'TIME' and 'PDCSAP_FLUX', and the initial parameters written in a text file:
- sim_fit_n: the number of simultaneous peaks subtracted from the original light curve after the fit. Value by default: 20.
- max_freq: the  maximum frequency of the studied domain. Value by default (c/d): 100 (considering delta Scuti stars).
- stop: the stop criterion can be the Signal to Noise Ratio (SNR) or the False Alarm Probability (FAP). Value by default: SNR.
- min_snr: limit value of the SNR for the peak to be significant. Value by default: 4 (Breger 1993).
- max_fap: limit value of the FAP for the peak to be significant. Value by default: 0.01 (Balona et al. 2014).
- os_ratio: oversampling ratio. Value by default: 5.
- tail_per: minimum value of the frequency for the tail of the periodogram, in order to calculate noise in every step of the pre-whitening. Value by default (c/d): 80 (considering delta Scuti stars). 

With every light curve, the code calculates the frequencies spectrum, or periodogram, with the Fast Lomb Scargle algorithm (Press & Ribicky 1989). It extracts the higher amplitude peak and evaluates if it is real signal or due to noise, either by the False Alarm Probability or by the Signal to Noise criterion, it is a decision of the user at the time of choosing the initial parameters. By default it is chosen to use as  stop criterion that S/N is greater than 4, (Breger 1993).
It fits frequency, amplitude and phase through non-linear optimization, using a multisine function. This function is redefined with the new calculated parameters. It does a simultaneous fit of a number of peaks (20 by default).
Then they are subtracted from the original signal and goes back to the beginning of the loop  with the residual, repeating the same process, until the stop criterion is reached. 
After that, the code can filter suspicious spurious frequencies, those of low amplitude below the Rayleigh resolution, and possible combined frequencies. 
This routine has been designed using the astropy packages https://www.astropy.org for the calculation of the periodograms and the lmfit packages https://lmfit.github.io/lmfit-py/ for the non-linear and simultaneous fitting of the extracted signals, using the non-linear least squares method for python.

# How to run it
- Copy the downloaded files MultiModes.py and ini.txt in your working directory.
- Copy in your working directory the directory with the light curves to be analysed.
- In in your working directory, type the command: run MultiModes.py --d name_of_directory_with_light_curves, if you use Anaconda or similar python enviroment, or type the command: python MultiModes.py --d name_of_directory_with_light_curves, if you use terminal unix. 
