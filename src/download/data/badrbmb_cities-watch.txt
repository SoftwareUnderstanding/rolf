# cities-watch
Monitoring urban development using satellite imagery; empirical validation of Zipf's law for cities (self-organised criticality).

# Description

Remote sensing offers the unique opportunity to monitor the evolution of human settlements from space.
 
Using multi-spectral images from Landsat 5, Landsat 7, Landsat 8 along with nighttime images from DMSP-OLS and NPP-VIIRS, this project aims to:

- Build a model quantifying the footprint of cities and monitor their evolution over time.
- Use the model to provide an empirical validation of the scaling law in the city size distribution, at difference scales (regional level, country level, worldwide).

Yearly images between 1992 to 2020 of Las Vegas, Nevada - USA
 
|RGB composites             |  Nighttime Lights        |
|:-------------------------:|:-------------------------:|
![Alt Text](./data/demo/las_vegas_area_rgb.gif) |  ![Alt Text](./data/demo/las_vegas_area_nl.gif)
|Image segmentation             |  Segmentation legend    |
|![Alt Text](./data/demo/las_vegas_area_preds.gif) | ![Alt Text](./data/demo/legend.png) |

# Examples

- Quantifying the urban growth of the Las Vegas area, Nevada - USA between 1992 and 2020*
![Alt Text](./data/demo/las_vegas_area_values.png)    
- Quantifying the  evolution of the 10 biggest urban agglomerations in India (by surface area) between 2015 and 2020*
![Alt Text](./data/demo/india_top10_cities.png)

- Empirical validation of the Zipf's law for cities in India, quantifying the surface areas of +15000 cities in the country, between 2015 and 2020* (Log-Log scales)
![Alt Text](./data/demo/zipf_law_india.png)
	- Read more on [Self-Organized Criticality and Urban Development](https://www.researchgate.net/publication/26531729_Self-Organized_Criticality_and_Urban_Development) from Batty and Xie, 1999.

_*2020 based on partial data as of May 2020_

_Important Note: computed areas for a specific city might differ from official reported numbers as this project quantifies the organic footprint of agglomerations as opposed to the administrative boundaries of cities._

# Methodology

## Satellite images

Satellite images are acquired and processed using Google Earth Engine. 
Details of implementation available in `cities_watch/image_utils.py`

### Day-time images

The landsat series ([Landsat 5](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C01_T1_SR), [Landsat 7](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C01_T1_SR) & [Landsat 8](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR)) of satellites allow a continuous multi-spectral observation of Earth' surface since 1984.

The processing steps to use these images in this project are:

- Getting all the surface reflectance images for each satellite
- Renaming all bands to a common names based on their wavelength

| Satellite / Bands       | blue | green | red | nir | swir1 | swir2 |
|------------------|------|-------|-----|-----|-------|-------|
| LANDSAT/LC08/C01 | B2   | B3    | B4  | B5  | B6    | B7    |
| LANDSAT/LE07/C01 | B1   | B2    | B3  | B4  | B5    | B7    |
| LANDSAT/LT05/C01 | B1   | B2    | B3  | B4  | B5    | B7    |

- Applying a cloud mask based on `pixel_qa` band for each image, which is a pixel quality attribute generated from the CFMASK algorithm.
- Adding a NDVI band to each image derived from the normalized difference of the NIR and Red bands
- Getting a yearly average composite by band of interest of all images in a image collection
- Resampling the yearly image bands to 250-meter pixels
	- This step is a compromise between speeding-up the future processing steps while still providing enough spatial resolution to de-contour city shapes.


### Night-time images

- The [DSMP-OLS](https://developers.google.com/earth-engine/datasets/catalog/NOAA_DMSP-OLS_NIGHTTIME_LIGHTS) detects visible and NIR emission sources at night with a dataset availability on Earth Engine between 1992 to 2014 on a monthly basis. 
- The [NPP-VIIRS](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG) Day/Night Band on earth engine provides monthly average radiance composite images using nighttime data between April 2012 to the present.

To allow the continuous observation of nighttime lights between 1992 to the present, this project implements a simple inter-calibration between DMSP-OLS and NPP-VIIRS, based on the range of dates in common between the two data sets between April 2012 and January 2014.

_Example of the inter-calibration step converting the DMSP-OLS 2013 median image over Las Vegas, Nevada (left) to a calibrated version (right).
The range of values of the right image a scaled linearly to be in-line with the radiance values from the NPP-VIIRS DNB band_
![Alt Text](./data/demo/inter_calibration_nl.gif)

## Model overview
To map the footprint of cities using both day-time (Landsat) and night-time (DMSP-OLS/NPP-VIIRS) images, a fully convolutional neural network was trained using the [Global Human Settlement Layer](https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2016_POP_GPW_GLOBE_V1) as a proxy for cities.

- Features stack, composed of 5 bands:
    - Red, Green and Blue bands from day-time images
    - NDVI composite, derived from the normalized difference of the NIR and Red bands from landsat images
    - Avg. Radiance band from the night-time images
- Training/testing data sets:
    - Yearly composite of the features stack for the year 2015, limited to selected locations in France.
- Target:
    - Original raster of the Global Human Settlement Layer for 2015, limited to selected locations.
    - The pixel values were converted to binary values assuming a city (class=1) has more than 50 people in a grid of 250x250m
- Model:
    - Unet model for image segmentation adapted from [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) 
    - A tensorflow model (trained and) hosted on Google AI Platform allows a smooth integration between earth engine images, conversion to tensors and prediction, and reassemble into earth engine data types.
Details of implementation available in `cities_watch/models.py`. [More information on Tensorflow & Earth Engine](https://developers.google.com/earth-engine/guides/tensorflow)
    

|![Alt Text](./data/demo/feature_stack_description.gif)


# Country-wise validation of the Zipf's law for cities' size distribution 

Given the size of country shapes, to visualize the scaling law in the city size distribution for a given country in a given year, the following additional resources are required:

- a bucket to store the shapes of the cities, vectorised from the image prediction, using `batch.Export.table.toCloudStorage` to run the computations server-side and avoid computation timeouts.

- a BigQuery GIS table to store the geo-tagged city shapes (with city name, area and rank)
Details of implementation available in `cities_watch/urban_mapper.py` & `cities_watch/urban_tagger.py`

Example of computed shapes for France, 2019 from BigQuery Geo Viz

![Alt Text](./data/demo/bigquery_france_shapes_2019.png)

Comparative results of Zipf's law for city size distribution in France and India, 2019. (Log-Log scales)
![Alt Text](./data/demo/zipf_law_compare.png)







   

