# Timage – A Robust Time Series Classification Pipeline

This is the accompanying repository to ICANN 2019 publication: Timage - A Robust Time Series Classification Pipeline.

A reprint of the paper can be found at [arXiv](https://arxiv.org/abs/1909.09149)

For maximum robustness and ease of use timage uses very little configuration.
To achieve this timage regards every time series as equal and transforms the time series to recurrence plots.
These plots are then used to train a deep Residual Network [ResNet](https://arxiv.org/abs/1512.03385). Same goes for inference.
This very simple, yet effective method works well for many different problems.

## Structure of this repository:
```bash
    .
    ├── README.md
    ├── code
    ├── figures
    └── results

```
* **./code** contains everything to recreate all experiments including a detailed README.md on how to use
  the timage software right away.
* **./results** contains pdf documents with results comparing timage to other systems evaluated on the [UCR archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) as printable versions and also relative comparisions on performance.
* **./figures** contains supplementary figures to make this README more appealing.

Please cite this work as:
```Latex
  @InProceedings{
        timage_2019,
        author       = {Marc Wenninger, Sebastian Bayerl, Jochen Schmidt, Korbinian Riedhammer},
        title        = {Timage -- A Robust Time Series Classification Pipeline},
        booktitle    = {Proceedings of ICANN 2019},
        year         = {2019},
        pages        = {10-18},
}

```
## Recurrence plots/ threshholded recurrence plots
In essence Reccurrence plots are a N-dim to 2-dim subspace mapping. 
This mapping is done by computing the point wise distances between all samples of a time series.
This always leads to a symetric matrix that can be plottet.
The formula describing this plots then applies a Heavside function that requries an Epsilon threshhold value.
<img src="figures/recurrenceplot.png" height=50>

We did not want to tune this hyper parameter so we plottet the pointwise distances after min-max scaling.
As an easy step to handle outliers we cut of distances larger than three times the standard deviation of all computed distances in the training set.

<img src="figures/unthresholded.png" with="300">\
<img src="figures/outlier.png" width="300">


### Examples of unthresholded Recurrence plots from the UCR archive
![Adiac Class B](figures/387.png "Class A from the Adiac dataset") ![Adiac Class B](figures/388.png "Class A from the Adiac dataset")

For more information on recurrence plots we recommend to visit http://recurrence-plot.tk/. 



# Results

## Accuracy of experiments on UCR 2015 datasets compared to other systems

| Dataset                        | WEASEL | F-t LSTM-FCN | ResNet50 SC| ResNet152 SC| ResNet50 AC| ResNet152 AC|
|--------------------------------|--------|--------------|----------|-----------|----------|-----------|
| Adiac                          |0.8312|**0.8849**|0.8440|0.8235|0.7238|0.6982|
| ArrowHead                      |0.8571|**0.9029**|0.8857|0.8629|0.8000|0.7486|
| Beef                           |0.7667|**0.9330**|0.7333|0.7667|0.8333|0.7000|
| BeetleFly                      |0.9500|**1.0000**|0.9000|0.9000|0.2000|0.2500|
| BirdChicken                    |0.8000|**1.0000**|0.9000|**1.0000**|0.1500|0.0500|
| Car                            |0.8667|**0.9670**|0.8833|0.9000|0.8500|0.8167|
| CBF                            |0.9833|**1.0000**|0.9044|0.9911|0.9478|0.9100|
| ChlorineConcentration          |0.7526|**1.0000**|0.7844|0.7852|0.7237|0.7109|
| CinCECGTorso                   |**0.9935**|0.9094|0.8913|0.8580|0.9080|0.8971|
| Coffee                         |**1.0000**|**1.0000**|**1.0000**|**1.0000**|**1.0000**|**1.0000**|
| Computers                      |0.6560|**0.8600**|0.7400|0.6960|0.2960|0.2080|
| CricketX                       |0.7641|**0.8256**|0.7359|0.7333|0.1385|0.1564|
| CricketY                       |0.7897|**0.8256**|0.7359|0.7538|0.6462|0.6410|
| CricketZ                       |0.7872|**0.8257**|0.7282|0.7615|0.0897|0.0923|
| DiatomSizeReduction            |0.8856|**0.9771**|0.9379|0.9346|0.8693|0.7941|
| DistalPhalanxOutlineAgeGroup   |0.7698|**0.8600**|0.7842|0.7842|0.0216|0.0432|
| DistalPhalanxOutlineCorrect    |0.7790|**0.8217**|0.8152|0.8080|0.2428|0.1014|
| DistalPhalanxTW                |0.6763|**0.8100**|0.7194|0.6835|0.0432|0.0144|
| Earthquakes                    |0.7482|**0.8261**|0.7770|0.7986|0.6331|0.6475|
| ECG200                         |0.8500|0.9200|0.8700|**0.9400**|0.7900|0.7900|
| ECG5000                        |**0.9482**|0.9478|0.9458|0.9442|0.9358|0.9307|
| ECGFiveDays                    |**1.0000**|0.9942|0.8165|0.9024|0.9477|0.8269|
| ElectricDevices                |0.7329|**0.7633**|0.7313|0.7297|0.6956|0.7181|
| FaceAll                        |0.7870|**0.9680**|0.7497|0.7828|0.7793|0.7704|
| FaceFour                       |**1.0000**|0.9772|0.7273|0.9091|0.8182|0.7273|
| FacesUCR                       |0.9522|**0.9898**|0.7771|0.8566|0.8371|0.8215|
| FiftyWords                     |**0.8110**|0.8066|0.7978|0.7868|0.7407|0.7253|
| Fish                           |0.9657|**0.9886**|0.9771|0.9771|0.9371|0.9257|
| FordA                          |0.9727|**0.9733**|0.9386|0.9235|0.6295|0.6667|
| FordB                          |0.8321|**0.9186**|0.8222|0.8074|0.5432|0.4926|
| GunPoint                       |**1.0000**|**1.0000**|**1.0000**|0.9933|0.9867|0.9867|
| Ham                            |0.6571|**0.8000**|0.7905|0.7429|0.7333|0.6952|
| HandOutlines                   |0.9487|0.8870|**0.9514**|0.9297|0.9162|0.9216|
| Haptics                        |0.3864|**0.5584**|0.5162|0.4968|0.4643|0.4351|
| Herring                        |0.6563|**0.7188**|0.6875|0.6563|0.5781|0.6094|
| InlineSkate                    |**0.6127**|0.5000|0.4036|0.4309|0.3055|0.3655|
| InsectWingbeatSound            |0.6404|**0.6696**|0.6177|0.6212|0.5944|0.5894|
| ItalyPowerDemand               |0.9514|**0.9699**|0.9602|0.9602|0.9475|0.9397|
| LargeKitchenAppliances         |0.6827|**0.9200**|0.8080|0.7573|0.6267|0.4987|
| Lightning2                     |0.5574|0.8197|**0.8525**|**0.8525**|0.3279|0.2295|
| Lightning7                     |0.7123|**0.9178**|0.8493|0.7945|0.3014|0.3425|
| Mallat                         |0.9655|**0.9834**|0.9326|0.9441|0.9365|0.9032|
| Meat                           |0.9167|**1.0000**|**1.0000**|0.9833|0.8500|0.8833|
| MedicalImages                  |0.7408|**0.8066**|0.7868|0.7803|0.7276|0.7184|
| MiddlePhalanxOutlineAgeGroup   |0.6039|**0.8150**|0.6364|0.6234|0.0000|0.0000|
| MiddlePhalanxOutlineCorrect    |0.8076|0.8333|0.8522|**0.8591**|0.1959|0.1134|
| MiddlePhalanxTW                |0.5390|**0.6466**|0.6234|0.5909|0.0195|0.0260|
| MoteStrain                     |0.9353|**0.9569**|0.8403|0.8866|0.4968|0.5719|
| NonInvasiveFetalECGThorax1     |0.9288|**0.9657**|0.9405|0.9405|0.9196|0.9288|
| NonInvasiveFetalECGThorax2     |0.9415|**0.9613**|0.9522|0.9547|0.9328|0.9405|
| OliveOil                       |**0.9333**|**0.9333**|**0.9333**|0.8667|0.5333|0.5000|
| OSULeaf                        |0.8967|**0.9959**|0.8678|0.8512|0.7355|0.6488|
| PhalangesOutlinesCorrect       |0.8170|0.8392|0.8578|**0.8590**|0.1643|0.2960|
| Phoneme                        |0.3265|**0.3602**|0.2410|0.2416|0.2152|0.1930|
| Plane                          |**1.0000**|**1.0000**|**1.0000**|**1.0000**|**1.0000**|0.9810|
| ProximalPhalanxOutlineAgeGroup |0.8488|**0.8878**|**0.8878**|0.8732|0.0049|0.0000|
| ProximalPhalanxOutlineCorrect  |0.8969|**0.9313**|0.9244|0.9278|0.3814|0.3883|
| ProximalPhalanxTW              |0.8098|**0.8275**|0.8049|0.8195|0.0634|0.0293|
| RefrigerationDevices           |0.5387|**0.5947**|0.5520|0.5440|0.4480|0.4267|
| ScreenType                     |0.5467|**0.7073**|0.4640|0.4720|0.2880|0.2853|
| ShapeletSim                    |**1.0000**|**1.0000**|0.5556|0.6333|0.4500|0.6667|
| ShapesAll                      |**0.9183**|0.9150|0.8850|0.8600|0.7600|0.7400|
| SmallKitchenAppliances         |0.7893|**0.8133**|0.7333|0.6880|0.6587|0.6853|
| SonyAIBORobotSurface1          |0.8236|**0.9967**|0.8802|0.9601|0.7953|0.7488|
| SonyAIBORobotSurface2          |0.9349|**0.9822**|0.8143|0.8458|0.7650|0.7870|
| StarLightCurves                |0.9773|0.9763|**0.9806**|**0.9806**|0.9709|0.9677|
| Strawberry                     |0.9757|**0.9864**|0.9811|0.9838|0.9676|0.9622|
| SwedishLeaf                    |0.9664|**0.9840**|0.9680|0.9616|0.9312|0.9232|
| Symbols                        |0.9618|**0.9849**|0.9719|0.9668|0.7739|0.8000|
| SyntheticControl               |0.9933|**1.0000**|0.7133|0.6700|0.6767|0.6767|
| ToeSegmentation1               |0.9474|**0.9912**|0.9167|0.9211|0.5570|0.5263|
| ToeSegmentation2               |0.9077|**0.9462**|0.9385|0.8846|0.4077|0.4692|
| Trace                          |**1.0000**|**1.0000**|**1.0000**|**1.0000**|**1.0000**|**1.0000**|
| TwoLeadECG                     |0.9982|**1.0000**|0.9895|0.9956|0.9816|0.9974|
| TwoPatterns                    |0.9898|**0.9973**|0.5157|0.5145|0.5028|0.4903|
| UWaveGestureLibraryAll         |0.9503|**0.9609**|0.9375|0.9436|0.9137|0.9112|
| UWaveGestureLibraryX           |0.8096|**0.8498**|0.7074|0.7004|0.4738|0.4693|
| UWaveGestureLibraryY           |0.7247|**0.7661**|0.7513|0.7133|0.5419|0.5034|
| UWaveGestureLibraryZ           |0.7700|**0.7993**|0.7289|0.7052|0.4913|0.4740|
| Wafer                          |**1.0000**|**1.0000**|0.9977|0.9966|0.9971|0.9942|
| Wine                           |0.8519|**0.8890**|0.6667|0.8333|0.5000|0.6667|
| WordSynonyms                   |**0.7241**|0.6991|0.6850|0.6771|0.6348|0.6176|
| Worms                          |0.8052|0.6851|0.8182|**0.8442**|0.4156|0.3506|
| WormsTwoClass                  |0.8052|0.8066|**0.8961**|0.8442|0.2857|0.2727|
| Yoga                           |0.9090|**0.9163**|0.9000|0.8827|0.8730|0.8713|

## Accuracy of experiments on new entries of UCR 2018 compared to the baseline(DTW)

| Dataset                  | DTW    | ResNet50 SC | ResNet152 SC | ResNet50 AC | ResNet152 AC |
|--------------------------|--------|----------|-----------|----------|-----------|
| ACSF1                    |0.6400|0.7800|**0.7900**|0.5700|0.5300|
| AllGestureWiimoteX       |**0.7157**|0.4943|0.5200|0.3757|0.3829|
| AllGestureWiimoteY       |**0.7286**|0.6000|0.5629|0.4500|0.4586|
| AllGestureWiimoteZ       |0.6429|**0.6514**|0.5871|0.5014|0.4943|
| BME                      |0.9000|**1.0000**|**1.0000**|0.9200|0.9533|
| Chinatown                |**0.9565**|0.7246|0.7565|0.0058|0.0087|
| Crop                     |0.6652|**0.7559**|0.7532|0.7405|0.7377|
| DodgerLoopDay            |0.5000|0.4875|**0.5125**|0.4000|0.3375|
| DodgerLoopGame           |**0.8768**|0.6812|0.6812|0.0000|0.0870|
| DodgerLoopWeekend        |0.9493|0.9420|**0.9710**|0.1667|0.0435|
| EOGHorizontalSignal      |**0.5028**|0.2790|0.2652|0.1160|0.1326|
| EOGVerticalSignal        |**0.4475**|0.2238|0.2569|0.1547|0.1215|
| EthanolLevel             |0.2760|**0.8400**|0.7100|0.7100|0.7100|
| FreezerRegularTrain      |0.8989|0.9961|**0.9975**|0.9221|0.9565|
| FreezerSmallTrain        |0.7533|**0.9793**|0.9354|0.0667|0.0368|
| Fungi                    |0.8387|0.8871|**0.9194**|0.8763|0.8548|
| GestureMidAirD1          |0.5692|0.5308|**0.6000**|0.4077|0.4231|
| GestureMidAirD2          |0.6077|**0.6231**|0.5538|0.4923|0.4846|
| GestureMidAirD3          |0.3231|0.4154|**0.4615**|0.3923|0.2231|
| GesturePebbleZ1          |0.7907|**0.9186**|**0.9186**|0.4070|0.3547|
| GesturePebbleZ2          |0.6709|**0.8544**|0.8354|0.0506|0.0949|
| GunPointAgeSpan          |0.9177|**0.9873**|0.9842|0.1582|0.0886|
| GunPointMaleVersusFemale |**0.9968**|0.9937|0.9937|0.2247|0.2658|
| GunPointOldVersusYoung   |0.8381|**0.9810**|**0.9810**|0.1873|0.2190|
| HouseTwenty              |**0.9244**|0.8319|0.8403|0.6807|0.6218|
| InsectEPGRegularTrain    |0.8715|**0.9719**|0.9639|0.8273|0.6988|
| InsectEPGSmallTrain      |0.7349|**0.9438**|0.8795|0.1004|0.2088|
| MelbournePedestrian      |**0.7906**|0.3604|0.3563|0.3298|0.3282|
| MixedShapesRegularTrain  |0.8416|**0.9645**|0.9509|0.8470|0.8276|
| MixedShapesSmallTrain    |0.7798|**0.9027**|0.8635|0.0247|0.0049|
| PickupGestureWiimoteZ    |0.6600|0.7400|**0.8000**|0.4800|0.4800|
| PigAirwayPressure        |0.1058|0.1442|**0.1683**|0.0913|0.1106|
| PigArtPressure           |0.2452|0.3510|**0.5288**|0.1635|0.2548|
| PigCVP                   |0.1538|0.4279|**0.5288**|0.3462|0.3221|
| PLAID                    |**0.8399**|0.8231|0.8119|0.7877|0.7858|
| PowerCons                |0.8778|0.9389|**0.9722**|0.9333|0.9000|
| Rock                     |0.6000|0.7800|**0.8400**|0.7400|0.6000|
| SemgHandGenderCh2        |**0.8017**|0.7867|0.7767|0.1900|0.2117|
| SemgHandMovementCh2      |**0.5844**|0.5244|0.5267|0.1422|0.1911|
| SemgHandSubjectCh2       |**0.7267**|0.6644|0.6889|0.3133|0.2578|
| ShakeGestureWiimoteZ     |0.8600|0.8800|**0.9400**|0.5400|0.6400|
| SmoothSubspace           |0.8267|**0.9933**|0.9867|0.9800|0.9533|
| UMD                      |**0.9931**|0.8264|0.7847|0.6944|0.6528|
