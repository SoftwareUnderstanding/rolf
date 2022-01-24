# Summarizing Radiology Reports using Neural Sequence to Sequence Learning


# Model -
  **Neural Sequence to Sequence Learning**
    
    An encoder decoder GRU-based architecture supplemented by an additional Attention based mechanism

# Motivation -
    1. Redundancy
    2. Factor of Human-Error
    3. A time consuming task
Refer to [this](https://www.ncbi.nlm.nih.gov/pubmed/22195100) report for additional information

# Reference
The idea is based on [this](https://arxiv.org/abs/1809.04698) paper 
    
# Dataset

A compilation of approx. 2700 unique reports collected from the [Open-i National Library of Medicine -  National Institues of Health](https://openi.nlm.nih.gov/)
    
# Vanilla Encoder-Decoder with GRU

    Validation split => 0.1
    Epochs => 30 
    Optimizer => Adam

![alt text](https://github.com/abhishekr7/report-summarizer/blob/master/IMG_20190406_153452.png)

# Results

The model performs well on a few test examples

    Input(Findings) : The lungs are clear . The heart and pulmonary XXXX are normal . Pleural spaces are clear . Mediastinal contours are normal . There is stable lucency in the right mid clavicle dating back to XXXX . 
    
    Output(Impression) : no acute cardiopulmonary disease . 
    
    
    Input(Findings) : The heart is normal in size . The mediastinum is stable . XXXX sternotomy changes are again noted . The lungs are clear of focal infiltrates . There is no pleural effusion . 
    
    Output(Impression) : no acute disease . 

But results in arbritrary sequences in some cases...

    Input(Findings) : Stable left chest cardiac XXXX generator with 2 distal leads in right atrium and right ventricle . Heart size normal . No pneumothorax , pleural effusion , or focal airspace disease . Emphysema . Stable calcified granulomas . Bony structures appear intact . 
    
    Output(Impression) : stable acute . acute . . acute cardiopulmonary . 
    
    
    Input(Findings) : Sternotomy sutures and bypass grafts have been placed in the interval . Both lungs remain clear and expanded with no infiltrates . Pulmonary XXXX are normal . 
    
    Output(Impression) : stable lungs . . . . . 
    
# Highlights

* Performs better without Teacher Forcing
* Attention does not provide any evident increase in the accuracy of the model 
* The model yields better results for shorter sequences of Findings
* No sign of overfitting observed (without using Dropout)
* Model would certainly perform better when trained on a larger dataset and for more epochs
