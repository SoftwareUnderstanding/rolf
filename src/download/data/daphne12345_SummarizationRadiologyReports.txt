# Automated Summarization of Breast Cancer Radiology Reports

<!---Breast cancer diagnosis is currently based on radiology reports written by humans. Manually summarizing the reports is time-consuming and leads to high text variability.
**This paper presents an automated summarization model of Dutch radiology reports using a combination of an encoder-decoder with attention and a separate BI-RADS score classifier (SVM)**. The summarization model was compared against a baseline model (encoder-decoder without attention) and performed 0.7\% better in ROUGE-L (50.8\% vs. 51.5\%). An accuracy of 83.3\% was achieved on the BI-RADS score classification. Additionally, a small qualitative evaluation with experts found the generated conclusions to be comprehensible and cover mostly relevant content, while their factual correctness is rather low. Overall, the developed model solves the summarization task well but some enhancements could improve the performance.--->

Breast cancer diagnosis is currently based on radiology reports written by humans. Manually summarizing the reports is time-consuming and leads to high text variability. **This project presents an automated summarization model of Dutch radiology reports using a combination of an encoder-decoder with attention and a separate BI-RADS score classifier (SVM) in Python using Tensorflow, Keras and SKLearn**. It contains notebooks for the summarization model (encoder-decoder with attention), baseline model (encoder-decoder without attention), and BI-RADS score classification (cancer severity score). 

## Example of a report containing the findings, the original and generated conclusion
![Image of an example report](Images/ex_report_translated.PNG)

## The summarization model setup
![Encoder-decoder model with Attention mechanism (Own diagram based on See et al. 2017 [1])](Images/hybrid_model.PNG "Encoder-decoder model with Attention mechanism (Own diagram based on See et al. 2017 [1]")
Encoder-decoder model with Attention mechanism (Own diagram based on See et al. 2017 [1]) <br/>
The <a href="https://github.com/thushv89/attention_keras">attention layer</a> used is Bahdanau's attention [2].

## Data
The used dataset includes roughly 50,000 breast cancer radiology reports from the Ziekenhuis Groep Twente (ZGT) hospital in Hengelo (Netherlands) recorded between 2012 and 2018. The reports are in Dutch and include data about clinical information, findings and conclusion. The clinical information and findings are treated as the input sequence. They contain the patient's medical history and result findings from the radiology procedures. This information usually indicates the breast cancer severity which is relevant for the conclusion.

The dataset is not provided in this repository as it contains confidential information.

## Running the notebooks
This repository is using Python >=3.7. 

1. Preprocessing: 
    -  Prerequisite: dataset with a similar structure of the one used here (Full text and target summaries split)
    -  Paths and columnnames need to be adjusted
    -  If using a different language than Dutch: Adjust language of stop words
    -  The entire notebook can be run and the preprocessed training, validation and test data will be saved in new files

2. Baseline:
    - Check if all necessary libraries are installed
    - The entire notebook can be run using the prepared data from step 1
    - The model will be trained and the results will be saved to an excel file
  
3. Hyperparameter tuning of the encoder-decoder model with attention:
    -  Check if all necessary libraries are installed
    -  The entire notebook can be run
    -  Performs hyperparameter tuning (the tested values might have to be adjusted)
    -  Saves the results of each combination to an excel file

4. BI-RADS score extraction and classification
    -  Can be run independently of step 2 and 3
    -  Uses the preprocessed data of step 1
    -  The strings used to extract the BI-RADS score in the conclusions might have to be adjusted to the data
    -  The entire notebook can be run
    -  The scores are extracted and used as labels
    -  Different classifiers are trained and tested on the data

## TODO (Future Work)
-  Apply model to an english data set (e.g. MIMIC III database)
-  Improve performance by testing small modifications on both models
-  Use a list of stopwords adapted to this task
-  Structuring of the findings in a preprocessing step
-  Structuring of the conclusions
-  Extract features (e.g. breast size) of the findings as labeled information
-  Check the resulting conclusions for grammar
-  Use a Deep Learning model for the classification task


## Paper
 Nguyen, E., Theodorakopoulos, D., Pathak, S., Geerdink, J., Vijlbrief, O., van Keulen, M., & Seifert, C. (2021). A Hybrid Text Classification and Language Generation Model for Automated Summarization of Dutch Breast Cancer Radiology Reports. In 2020 IEEE Second International Conference on Cognitive Machine Intelligence (CogMI) (pp. 72-81). [9319371] IEEE. https://doi.org/10.1109/CogMI50398.2020.00019


## References
<a id="1">[1]</a> 
A. See, P. J. Liu, and C. D. Manning, <br />
“Get to the point: Summarization with pointer-generator networks” <br />
2017. <br />
[Online]. Available: https://arxiv.org/pdf/1704.04368

<a id="2">[2]</a>
D.  Bahdanau,  K.  Cho,  and  Y.  Bengio,  “Neural  machine  translationby  jointly  learning  to  align  and  translate,” <br />
2014.  <br />
[Online].  Available:https://arxiv.org/pdf/1409.0473
