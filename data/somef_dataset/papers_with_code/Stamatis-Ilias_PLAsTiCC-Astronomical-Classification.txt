
# Introduction

## Data

The data that was used in this project came from the LSST dataset, an astronomical dataset made by  values based on the survey data of the telescope, that when analyzed will help classify the astronomical data that the telescope will collect in the future.

The Data consists of a train set and a test set, containing 7848 and 3490000 objects respectively, making the train to test size ratio rather small. This was the biggest challenge that we faced in this competition, and it was an understandable hurdle regarding the nature of the data. Precision was the goal of this competition and we believe we did a rather good job in that area.

##  Equipment Used

Due to the huge size of the test set **20GB** we used a google console cloud linux system, equiped with 56GB of RAM and a 10-core processor, to help do the computations as fast as possible. Even so, many ideas we had in mind didn't have the time to be implemented for the competition but we will describe them briefly later.

## Data Analysis and Study

The data was divided into two sets. One for the **timeseries data** of each object, that contained the *Julian Dates, the flux and the corresponding passband* of each observation, while also having the *flux error* of the observation and a *detected* value that was equal to 1 if the object's brightness is significantly different at the 3-sigma level relative to the reference template.



```python
import pandas as pd
df = pd.read_csv("../all/training_set.csv")
df_meta = pd.read_csv("../all/training_set_metadata.csv").set_index("object_id")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>object_id</th>
      <th>mjd</th>
      <th>passband</th>
      <th>flux</th>
      <th>flux_err</th>
      <th>detected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>615</td>
      <td>59750.4229</td>
      <td>2</td>
      <td>-544.810303</td>
      <td>3.622952</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>615</td>
      <td>59750.4306</td>
      <td>1</td>
      <td>-816.434326</td>
      <td>5.553370</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>615</td>
      <td>59750.4383</td>
      <td>3</td>
      <td>-471.385529</td>
      <td>3.801213</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>615</td>
      <td>59750.4450</td>
      <td>4</td>
      <td>-388.984985</td>
      <td>11.395031</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>615</td>
      <td>59752.4070</td>
      <td>2</td>
      <td>-681.858887</td>
      <td>4.041204</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The other data file contains the metadata of each star, differentiating the stars by an object_id.



```python
df_meta.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ra</th>
      <th>decl</th>
      <th>gal_l</th>
      <th>gal_b</th>
      <th>ddf</th>
      <th>hostgal_specz</th>
      <th>hostgal_photoz</th>
      <th>hostgal_photoz_err</th>
      <th>distmod</th>
      <th>mwebv</th>
      <th>target</th>
    </tr>
    <tr>
      <th>object_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>615</th>
      <td>349.046051</td>
      <td>-61.943836</td>
      <td>320.796530</td>
      <td>-51.753706</td>
      <td>1</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>0.017</td>
      <td>92</td>
    </tr>
    <tr>
      <th>713</th>
      <td>53.085938</td>
      <td>-27.784405</td>
      <td>223.525509</td>
      <td>-54.460748</td>
      <td>1</td>
      <td>1.8181</td>
      <td>1.6267</td>
      <td>0.2552</td>
      <td>45.4063</td>
      <td>0.007</td>
      <td>88</td>
    </tr>
    <tr>
      <th>730</th>
      <td>33.574219</td>
      <td>-6.579593</td>
      <td>170.455585</td>
      <td>-61.548219</td>
      <td>1</td>
      <td>0.2320</td>
      <td>0.2262</td>
      <td>0.0157</td>
      <td>40.2561</td>
      <td>0.021</td>
      <td>42</td>
    </tr>
    <tr>
      <th>745</th>
      <td>0.189873</td>
      <td>-45.586655</td>
      <td>328.254458</td>
      <td>-68.969298</td>
      <td>1</td>
      <td>0.3037</td>
      <td>0.2813</td>
      <td>1.1523</td>
      <td>40.7951</td>
      <td>0.007</td>
      <td>90</td>
    </tr>
    <tr>
      <th>1124</th>
      <td>352.711273</td>
      <td>-63.823658</td>
      <td>316.922299</td>
      <td>-51.059403</td>
      <td>1</td>
      <td>0.1934</td>
      <td>0.2415</td>
      <td>0.0176</td>
      <td>40.4166</td>
      <td>0.024</td>
      <td>90</td>
    </tr>
  </tbody>
</table>
</div>



## Target Distribution

The goal of the challenge is to classify each object to a target, based on features we will extract from the above data. Even though a simple classification problem sounds easy, the fact that the data ratio is skewed is not the only problem that prevents the problem from being simple. Looking at the graph below ***Taken from Study.ipynb***, we can see that the distribution of the targets are not equal.

<img src="img/targets.png">

For this reason, when classifying the Data we preproccessed it with **SMOTE**, which is a certain algorithm for providing the training set with more data, as to stabilize the disparities between the unequal size classes.

*For more on SMOTE you can visit this link  https://arxiv.org/pdf/1106.1813.pdf*

## Viewing the data and inferring our next move
### Autoencoded time-series data

By plotting a random flux 

<img src = "img/flux_example.png">

We can see that it is divided in 3 to 4 different time ranges, making the study of such a timeseries harder than normal as it is missing values in certain places. To compat this, we used an Autoencoder later in the challenge, that was train on the data and made it's own normalized and time invariant data. This could help in 2 ways
1) Extract Random Features that the autoencoder found useful
2) Use the encoded model's data to re-train the model

The first helped our results quite a lot, while the second was one of those things that we didn't have the time to finish, as the autoencoded model just for the training set upped its size by 200%. (And that's because it was filling gaps in the data). This means that the test set variant would be quite huge.

The code for both of these can be found in *python_files_test/autoenc.py, python_files_test/encoder.py*

### Dividing the flux

Another way we tried to compat the above was, by dividing the flux series in 3-4 pieces based on a threshold and then applying feature extraction in these places seperately. This worked great in the training set, but failed to work in the test set cause we didn't took into account that the data there was seperated a lot differently. The code for the above is given below:<br>

```python

def sxedio5(series):
    threshold = 120 #Threshold found as the minimum of all the "dead-time-spaces"
    diffs = np.diff(series.mjd)
    diff_ids = (np.where(diffs>threshold))
    print(diff_ids)
    
    if(len(diff_ids[0])==2):
        series1 = series[:diff_ids[0][0]]
        series2 = series[diff_ids[0][0]:diff_ids[0][1]]
        series3 = series[diff_ids[0][1]:]
        agg1 = series1.agg(aggs_all)
        agg2 = series2.agg(aggs_all)
        agg3 = series3.agg(aggs_all)
        aggs = pd.concat([agg1, agg2, agg3], axis=1)
        
    elif(len(diff_ids[0])==3):
        series1 = series[:diff_ids[0][0]]
        series2 = series[diff_ids[0][0]:diff_ids[0][1]]
        series3 = series[diff_ids[0][1]:diff_ids[0][2]]
        series4 = series[diff_ids[0][2]:]
        agg1 = series1.agg(aggs_all)
        agg2 = series2.agg(aggs_all)
        agg3 = series3.agg(aggs_all)
        agg4 = series4.agg(aggs_all)
        aggs = pd.concat([agg1, agg2, agg3, agg4], axis=1)
        
    else:
        mjd_diff = np.max(series.mjd) - np.min(series.mjd)
        mjd_diff_div =int(np.ceil(mjd_diff/4))
        #print(mjd_diff, mjd_diff_div)
        series1 = series[:mjd_diff_div]
        series2 = series[mjd_diff_div:2*mjd_diff_div]
        series3 = series[2*mjd_diff_div:3*mjd_diff_div]
        series4 = series[3*mjd_diff_div:]
        agg1 = series1.agg(aggs_all)
        agg2 = series2.agg(aggs_all)
        agg3 = series3.agg(aggs_all)
        aggs = pd.concat([agg1, agg2, agg3], axis=1)
        
    if(aggs.shape == (6,9)):
        agg1[:] = 0
        #print(agg1)
        aggs = pd.concat([aggs,agg1],axis=1)
    return aggs   
```

Because the above failed to work in the test-set, we just did statistical analysis to the set without any further preprocessing.

## How we Tackled the problem

In *Study.ipynb* we plotted 20 fluxes for each target class to see how they compare, and by that analysis we got most of the ideas on how to tackle the classification problem.

<img src="img/flux_target.png">

We can see that same-class objects look a lot like each other, as they have the **same peak-patterns** the **same passband-patterns** and sometimes **same times of observations.**

For that reason, we extracted the following statistical features for each object_id, by sorting for each different passband and by not.<br>
*Most of the work for Feature Extraction, can be found in the kernel Feature Extraction_neg.ipynb and in the file agg_test.py, the code is given below*
<br>

```python
def process_flux(df):
    # Get the squared ratio of the flux and flux_err as a feature
    flux_ratio_sq = np.power(df['flux'].values / df['flux_err'].values, 2.0)

    df_flux = pd.DataFrame({
        'flux_ratio_sq': flux_ratio_sq, 
        'flux_by_flux_ratio_sq': df['flux'].values * flux_ratio_sq,}, 
        index=df.index)
    
    return pd.concat([df, df_flux], axis=1)


def process_flux_agg(df):
    # Create more flux features by the statistical deviations between the flux, its error and the squared
    # error from before
    flux_w_mean = df['flux_by_flux_ratio_sq_sum'].values / df['flux_ratio_sq_sum'].values
    flux_diff = df['flux_max'].values - df['flux_min'].values
    
    df_flux_agg = pd.DataFrame({
        'flux_w_mean': flux_w_mean,
        'flux_diff1': flux_diff,
        'flux_diff2': flux_diff / df['flux_mean'].values,       
        'flux_diff3': flux_diff /flux_w_mean,
        }, index=df.index)
    
    return pd.concat([df, df_flux_agg], axis=1)

def process_meta(filename):
    meta_df = pd.read_csv(filename)
    
    meta_dict = dict()
    # distance
    meta_dict.update(haversine_plus(meta_df['ra'].values, meta_df['decl'].values, 
                   meta_df['gal_l'].values, meta_df['gal_b'].values))
    #
    meta_dict['hostgal_photoz_certain'] = np.multiply(
            meta_df['hostgal_photoz'].values, 
             np.exp(meta_df['hostgal_photoz_err'].values))
    
    meta_df = pd.concat([meta_df, pd.DataFrame(meta_dict, index=meta_df.index)], axis=1)
    return meta_df

# General aggregates
aggs_all = {
        'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'detected': ['mean'],
        'flux_ratio_sq':['sum', 'skew'],
        'flux_by_flux_ratio_sq':['sum','skew'],
    }

aggs_pb = {
        'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'flux_ratio_sq':['sum', 'skew'],
        'flux_by_flux_ratio_sq':['sum','skew'],
    }

# Aggregates for tsfresh. Tsfresh is a python modeling from timeseries feature extraction!
fcp = {
        'flux': {
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,
            'mean_change': None,
            'mean_abs_change': None,
            'length': None,
        },
                
        'flux_by_flux_ratio_sq': {
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,       
        },
                
        'flux_passband': {
            'fft_coefficient': [
                    {'coeff': 0, 'attr': 'abs'}, 
                    {'coeff': 1, 'attr': 'abs'}
                ],
            'kurtosis' : None, 
            'skewness' : None,
        },
                
        'mjd': {
            'maximum': None, 
            'minimum': None,
            'mean_change': None,
            'mean_abs_change': None,
        },
    }
```

## Our own features

In most of the files we have implemented our own algorithms for extracting features, which are the following.<br>
1) Find the number of local maximas in a flux curve.<br>
2) Find the Dynamic Time Wrapping (DTW) between each passband for one curve (meaning 15 features for all of 6 passbands).<br>
3) Find the first time changes for each curve (derivation).<br>
4) Find the longest time that the flux keeps falling, and calculate the peak time interval.<br>

### Metadata features<br>
In the data we have two kinds of values that help find the host galaxy of the star, and these are **hostgal_specz** and (**hostgal_photoz, hostgal_photoz_err**). The first of these values, is of greater importance and could help our work greatly, but it is not available in the whole of the test set. For this, we tried to fit the hostgal_photoz of each star and it's distmod to the hostgal_specz, and obtained the following equation that we used to calculate this new feature: <br>

```python     
meta_df["hostgal_photoz_certain_mine"] = -6.09*(10**(-14))*np.exp(0.6713*meta_df["distmod"])+0.04902+meta_df["hostgal_photoz"]
```

For what concerns the distances of each star, we didn't do much as they are universal for all classes 
<img src="img/position_example.png">

Even so we used the following haversine function to compare the galactic and earthly distances of the two for any diviations

```python
def haversine_plus(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) from 
    #https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    #Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    #Implementing Haversine Formula: 
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),  
                          np.multiply(np.cos(lat1), 
                                      np.multiply(np.cos(lat2), 
                                                  np.power(np.sin(np.divide(dlon, 2)), 2))))
    
    haversine = np.multiply(2, np.arcsin(np.sqrt(a)))
    return {
        'haversine': haversine, 
        'latlon1': np.subtract(np.multiply(lon1, lat1), np.multiply(lon2, lat2)), 
   }
```

## Other hurdles and things that could have been implemented

Another problem with the data, that didn't help with the classification was that *target 99* was absent from the test data, for this reason we had to calculate the probability of one object being there based on the weighted probabilities of the other classes, and for that reason we used the following function.<br>

```python
# Get the median, mean and max of the other probabilities and infer the  probability of class 99 based on these
def GenUnknown(data):
    return ((((((data["mymedian"]) + (((data["mymean"]) / 2.0)))/2.0)) + (((((1.0) - (((data["mymax"]) * (((data["mymax"]) * (data["mymax"]))))))) / 2.0)))/2.0)

feats = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
         'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
         'class_92', 'class_95']

y = pd.DataFrame()
y['mymean'] = preds_df[feats].mean(axis=1)
y['mymedian'] = preds_df[feats].median(axis=1)
y['mymax'] = preds_df[feats].max(axis=1)

class_99 = GenUnknown(y)
preds_df["class_99"] = class_99
```

### Reading the test set
For doing this we used a library created by an other competitor, that can be found in the read_test_files folder

### Gaussian Process
Maybe the only thing that we didn't have time to implement and could improve our score for the best was to fit Gaussian Processes to the time-series data, so we could infer statistical features for every different set of observations. A gaussian process is a set of functions that can fit data without using any parameters. This means that we can feed any curve in a good algorithm and it will return the posterior parameters best suited for the fit.
<br>
*For more on GP visit this link http://katbailey.github.io/post/gaussian-processes-for-dummies/*
<br>

Generally this is a time consuming problem, and for that we have to constraint the function to only fit processes of certain range, curvature, noise and other such characteristics.





# Deep Learning

In the file **Deep Learning.ipynb** we used a simple LGB Random Forest to make predictions on the data.<br>
The reason that we used it besides it generally giving the best predictions from all other models (RNN, LSTMNN, Machine Learning tasks), was because of its speed. The speed in this competition was key, and by using this algorithm that could train in less than a minute for 10-fold CV we managed to save a lot of much needed time, for our other calculations


```python

```
