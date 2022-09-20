# ComEst: a Completeness Estimator of Source Extraction on Astronomical Imaging

Authors: 
[I-Non Chiu](mailto:inonchiu@usm.lmu.de), [Shantanu Desai](mailto:shantanu@usm.lmu.de), [Jiayi Liu](https://scholar.google.com/citations?hl=en&user=yk1ivyoAAAAJ)

**ComEst** is a completeness estimator of CCD images conducted in astronomical observations saved in the [FITS](http://fits.gsfc.nasa.gov/fits_documentation.html) format. Specifically, **ComEst** is designed for estimating the completeness of the source finder **[SExtractor](http://www.astromatic.net/software/sextractor)** on the optical and near-infrared (NIR) imaging of point sources or galaxies. The completeness is estimated as the detection rate of simulated sources-- simulated by the python image simulation toolkit **[GalSim](https://github.com/GalSim-developers/GalSim)**-- which are injected into the observed images with various configuration. In this way, **ComEst** estimates the completeness of the source detection-- as a function of flux (or magnitude) and, moreover, position on the CCD-- directly from the image itself. **ComEst** only requires the observed iamge as the input and performs the end-to-end estimation of the completeness. Apart from the completeness, **ComEst** can also estimate the purity of the source detection. 

**ComEst** is released as a **Python** package with an easy-to-use and flexible syntax. More information can be found in the paper published in [Astronomy and Computing](http://adsabs.harvard.edu/abs/2016A%26C....16...79C).

If you use **ComEst** in your work please contact the authors.


### Installation

 1. Download **ComEst**.

 2. To Install **ComEst** one simply types
    
    ```
    python setup.py install
    ```
    
    If you dont have the permission to install, you can instead type
    
    ```
    python setup.py build
    ```
    
    and then add the library (i.e., `build/lib`) to your `PYTHONPATH`.
 
 Or you can simply type:
   ```
   pip install git+git://github.com/inonchiu/ComEst.git
   ```
 
 3. To run **ComEst** one needs the following prerequisites.

  - **[Numpy](http://www.numpy.org/)** 
  - **[SciPy](http://www.scipy.org/)**
  - **[PyFITS](http://www.stsci.edu/institute/software_hardware/pyfits/Download)**
  - **[GalSim](https://github.com/GalSim-developers/GalSim)** 
  - **[SExtractor](http://www.astromatic.net/software/sextractor)**

    **ComEst** is tested against numpy > v1.9.1, SciPy > v0.14.0, PyFITS > v3.3, GalSim > v1.3 and SExtractor > v2.19, although **ComEst** should work with the older versions of prerequisites. Since **ComEst** heavily relies on **GalSim**, we strongly recommend that users should install **GalSim** with version higher than v1.3.

 4. The installation is done. Now you can load in **ComEst** package by typing the following in python environment 
    
    ```
    import comest
    ```

### Documentation

Please see [here](http://www.usm.uni-muenchen.de/people/inonchiu/ComEst/index.html). IMPORTANT: this documentation is out-dated. Yes, I know, I need to find time to update it...

### Acknowledgement

This work is dedicated to Chien-Ho Lin（林千禾）in Taiwan. 

We acknowledge the support by the DFG Cluster of Excellence "Origin and Structure of the Universe", the DLR award 50 OR 1205 that supported I. Chiu during his PhD project, and the Transregio program TR33 "The Dark Universe". 
The computations for **ComEst** have been have been carried out on the computing facilities of the Computational Center for Particle and Astrophysics (C2PAP) and of the Leibniz Supercomputer Center (LRZ).


### Contribution

Any comment and suggestion is welcome, please email to [I-Non Chiu](mailto:inchiu@asiaa.sinica.edu.tw), [Shantanu Desai](mailto:shantanud@iith.ac.in) or [Jiayi Liu](mailto:astro.jiayi@googlemail.com).

