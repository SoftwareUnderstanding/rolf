# Machine-Estimation-of-Exposure---Massdep
This repository contains the end-to-end process in building a classification algorithm to classify exposures from immediate response action reports

# Motivation
Massachusetts Department of Environmental Protection (MassDEP) is an agency of the commonwealth of massachusetts which ensures the timely cleanup of land, water and air. Immediate Response Reports are mandatory risk reduction measure taken at sites which have been impacted by sudden chemical releases or conditions requiring rapid notification to the MassDEP.

The officials at MassDEP summarize these reports into a preliminary action form which contains 5 flag questions each concerning one media, namely, groundwater, public water supply, private water supply, monitoring well and indoor air. If any of these media has been affected, the condition is classified as an exposure and needs to be revisited. 

The project seeks to classify these reports into exposures or not exposure understanding and learning patterns in the immediate response action reports. 

# Workflow

## 1. Data Crawler

The immediate response action reports are present on the MassDEP website and the project built a data crawler to pull all these reports on the local machine. 

## 2. Data Cleaning 

### 1.Text Cleaning
 - Replacing
 
    Items | Example
    ---|---
    Abbreviations | "Immediate Response Action (IRA)", <br>"Massachusetts Department of Environmental Protection (DEP)", <br>"Substantial Release Migration (SRM)"
    Addresses | "218 South Street in Auburn, Massachusetts", <br>"218 South Street, Auburn MA 01501"
    Dates and Time | "June 2017", <br>"June 16, 2017" <br>"9:57 a.m.", <br>"On June 13 & 14, 2017"
    **Numbers** | ***
    Longitudes and Latitudes | "42o10'36" north latitude (42.17650 °N), 71o50'19" east longitude (-71.83856 °W)" 
    Regulation and Forms | "310 CMR 40.0420(7)", <br>"Forms BWSC 123"
    Measurements | "95,832 square feet (approximately 2.20 acres)", <br>"approximately 50' south of the release area", <br>"6-7' below grade", <br>"<10 ppmv", <br>"within 1⁄2-1' of the water table"
    RTN Numbers | "RTN 2-20220"
    Chemicals | 
    Legal Terms Quotes | "‘significant risk’", <br>"“level of diligence reasonably necessary to obtain the quantity and quality of information adequate to assess”"
    Names | "Robert L. Haroian"
    
 - Removing
    * Tables and forms within text part
    * Footnotes
    * Captions within text part
    * Special characters ("&", and so on)

## 3. Load text data as python dataframe

Load the positive and negative data in the separate folders as one dataframe for further processing

## 4. Count Vectorizer and TFIDF 

Count Vectorizer converts a collection of text documents to a matrix of token counts

TFIDF Vectorizer converts a collection of text documents to a matrix of TF-IDF features.

## 5. SMOTE 

An approach to the construction of classifiers from imbalanced datasets is described. A dataset is imbalanced if the classification categories are not approximately equally represented. Please find more information here : https://arxiv.org/pdf/1106.1813.pdf

## 6. Machine Learning

1. Logistic Regression
2. Naive Bayes Classifier
3. Random Forest
4. XG Boost

Documenting only logistic regression and naive bayes as the two algorithms gave the best results along with time efficiency.

## 7. Results

The results were evaluated on the basis of precision and recall. We look at both the metrics with a view of the type of implementation in future work.

# 8. Running the codes 

If you are using the jupyter notebook, please install all the libraries as prompted. You can install any library using the anaconda prompt 

```
pip install library
```

