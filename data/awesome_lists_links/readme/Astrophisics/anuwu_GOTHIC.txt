# DAGN-Blindtest

![title](Images/ngc-3758_background.jpg)

Welcome! The galaxy centred in the image above is NGC-3758. It's a Double Active Nuclei Galaxy (**DAGN**) and boasts of two supermassive black-holes in its heart. Such galaxies are hard to find and until date, have only been found by chance/manual observation. The aim of this project is automate searching the Sloan Digital Sky Survey (**SDSS**) catalog to find galaxies that look similar to NGC-3758

**Check out our preliminary [discoveries](#discoveries)!**

# Why?

- The already existing list of DAGNs is miniscule. Having a large catalog of DAGNs will benefit astrophysicists. Surely there are more out there waiting to be found.
  > [Gimeno et. al.](https://iopscience.iop.org/article/10.1086/421371/pdf) has about 50 valid galaxies in SDSS
- There are 208 million galaxies in the SDSS catalog alone. Manual observation is not an option as it's slow and erroneous. A computer can resolve details which a human may be unable to.

[//]: # (Add link about )

## How do I conduct my own grand search?

- Clone this repository to your local directory or your google drive. (*It's recommended to do this on drive as the classification process requires FITS images to be downloaded from SDSS, which will consume enormous amounts of network data*)
- Obtain a list of galaxies to classify. This must be in `.csv` format with the following schema -
```
objID,ra,dec
1237650761852846175,178.154976439536,-2.46941711974289
1237651539239895057,139.183424654422,59.7746157484119
.
.
1237668670253564040,155.943139006441,60.7520145674758
```
- In the `Batches` folder, create another folder and put your `.csv` file in it. Ensure that the name of the folder and the file are identical
- Run the `DAGN-Blindtest.ipynb` notebook. It is self-explanatory.

  > If you're running it locally, you may skip the code cells under the **Preparations** heading in the notebook. Check out the [package requirements](#requirements)

- To run GOTHIC locally for a batch of galaxies, use the following command (replace `<batchname>` with the name of your batch -
```
python gothic.py <batchname>
```

## Interpreting the results

Classification result will be available in the folder which contains the `.csv` file. A sample entry is shown below -
```
objID,u-type,u-peaks,g-type,g-peaks,r-type,r-peaks,i-type,i-peaks,z-type,z-peaks
1237666301628121372,NO_PEAK,"[]",NO_PEAK,"[]",NO_PEAK,"[]",SINGLE,"[(44, 38)]",DOUBLE,"[(38, 83)(50, 51)]"
```
An object is analysed in all 5 bands (*u, g, r, i, z*) of SDSS. Its classified type, along with the relative location (*with respect to cutout image*) of the ascertained galactic nuclei, is saved. In the example above, the galaxy `1237666301628121372` has no detected galactic nuclei in the u, g and r-bands. However, it's a single nuclei in the~ i-band and double in the z-band.
> Z-band images are noisy for our purposes, they are mainly to be treated as false positives

## Requirements

The following packages are required to be able to run locally (*works for python `< 3.8`*) -
 - `numpy`
 - `astropy`
 - `scipy`
 - `pandas`
 - `matplotlib`
 - `bs4` (*AKA - BeautifulSoup*)
 - `lxml` (*Needed by BeautifulSoup for scraping SDSS*)
 - `opencv-python` (*AKA - OpenCV*)

# How?

The FITS frames of an object is scraped off SDSS and a cutout of `40''` is made centred around the object. The process is demonstrated with the r-band image of the galaxy MRK-789 from the [Gimeno](https://iopscience.iop.org/article/10.1086/421371/pdf) catalog

### 1. Smoothening

The image is scaled appropriately and smoothed with a gaussian filter to remove random noise. Below is MRK-789 after smoothening

![alt-text](Images/smooth.png)

### 2. Edge Detection with Convex Hull

[Canny](https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html) is applied to find the edges in this object. Since the edges reported are sporadic and do not properly enclose the object, the convex hull is taken as a bound for the galaxy in the image.

![alt-text](Images/hull.png)

As can be seen, the hull can end up enclosing stray objects (*bottom left*) in the image and also cover considerable area with barely any signal

### 3. Fitting intensity profile

The intensity profile is fit to a [Sersic](https://en.wikipedia.org/wiki/Sersic_profile) light profile, from which the noise/signal levels in the image are inferred. This helps in determining which region in the hull contains significant intensity
> (*More info on Sersic profile [here](https://arxiv.org/pdf/astro-ph/9309013.pdf)*)

![alt-text](Images/signal.png)

### 4. Stochastic Hill Climbing

Hill climbing is performed in the search region with a random initial seed point. This is done `> 100` times and a list of optima is created.

![alt-text](Images/hill-opts.png)

As can be seen, an optima is also marked in the stray object. It also happens to be brighter than the central object. This needs to be weeded out.

### 5. Peak Filtration

Depth-first-search is performed in the search region to determine the disjoint patches of light. Based upon a criteria on the size of the patches, and its distance from the centre of the image, the true galactic nuclei are inferred. In our case, the galaxy happens to be DAGN.

![alt-text](Images/peaks.png)

# Discoveries

We have found about 25, previously undiscovered, confirmed double-nuclei galaxies using this pipeline. Here are some of them, and there are more to come!

[1237657190370247192](http://skyserver.sdss.org/dr15/en/tools/explore/summary.aspx?id=1237657190370247192) --> ![alt-text](Images/discovery1.png)

[1237678437017977032](http://skyserver.sdss.org/dr15/en/tools/explore/summary.aspx?id=1237678437017977032) --> ![alt-text](Images/discovery2.png)

[1237666339726230221](http://skyserver.sdss.org/dr15/en/tools/explore/summary.aspx?id=1237666339726230221) --> ![alt-text](Images/discovery3.png)
