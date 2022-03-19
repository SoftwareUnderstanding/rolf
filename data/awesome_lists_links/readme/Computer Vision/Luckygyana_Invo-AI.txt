# Invo-AI: An automatic E-Invoicing using Image Processing
We propose an extraction system that use knowledge of the types of the target fields to generate extraction candidates, and a neural network architecture that learns a dense representation of each candidate based on neighbouring words in the document. These learned representations are not only useful in solving the extraction task for unseen document templates from two different domains, but are also interpretable in classic document processing.

### Our Project: [Source Code](https://github.com/Luckygyana/Invo-AI) | [Youtube]() | <br>

### AUTHOR CONTRIBUTIONS
> Subject: &nbsp; &emsp; &emsp; &nbsp;**Invo-AI** <br>
> Topic:&emsp; &emsp; &nbsp; &nbsp; &nbsp; &nbsp;**Electronic Invoicing using Image Processing** <br>
> Assignment: &emsp;  **Source Code (Final Presentation)** <br>
> Authors:&ensp; &ensp; &emsp; &nbsp; **GYANENDRA DAS** <br>
> Date: &ensp; &ensp; &emsp; &emsp; &nbsp; July 8,2020  <br>
<br>

## Developing an Automatic Invoice Extraction system using Python <br>

# Features!

  - Extract Information from Scanned Invoices to a XML file
  - Multilanguage Support

<figure>
  <img src="./Results/OCR_Text_Parsing.gif" align="right" width=450/>
</figure>

### Our Approach

Our complete Model works in the following eight steps:

* Convert PDF to JPG
* Detecting Bbox for all Text
* Bbox_mapper extract contours, sort them in all manners and extract text from them.
* Recognition of Text using OCR-Tesseract LSTM
* Ensemble Searching the keyword to locate Table Header
* Segregate the Image into Info (Non-Table Part) and Sheet (The item Table)
* Direct filling the value of Sheet in the XLS file. 
* Searching Key to extract Info values and mapping it in the XLS File.

#### Convert PDF to JPG

For This Task We use  [https://pypi.org/project/pdf2image/]

#### Detecting Bbox for All Text

- Binary images in ExtractStructure class for image processing

*Bbox_mapper* extract contours, sort them in all manners and extract text from them in sequential Order

<figure>
  <img src="./Results/Table_Detection_Algorithm_Demo.gif" align="right" width=300/>
</figure>

<br>

Optimization Techniques incorporated to extract the Grid Structure Class with higher accuracy involves:
- Gaussian Blur
- getStructuringElement (to get Kernel size)
- Dilate
- Erode
- Convolution

This is main task in overall Process. For this task we applied Three methods.

* Use Pytesseract [https://pypi.org/project/pytesseract/]
    * Advantage : Every Word is Detecting and creating one and more bbox per an word
    * Disadvantage : It cannot be able to detect Semantic pair with one bbox like Invoice no and its value is in different bbox
* Use EfficientDet [https://arxiv.org/abs/1911.09070]
    * Advantage : We tried to divide the image to some classes like Shipping, Buying, Header,Footer, Table. 
        * For this task we used our own labeled dataset of around 2500 images.
        * With a good training pipeline effdet d5 we able to acheive good loss of 0.83
        * We added WBF [https://arxiv.org/abs/1910.13302] and get loss of 0.42
    * Disadvantage : We cannot detect line and words because lack of data 
* Use CRAFT Model [https://arxiv.org/pdf/1904.01941.pdf]
    * Advantage : We can detect the lines and semantic pair both adjusting the best threshold
    * Disadvantage : There is no disadvantage but model should need to be optimized

![combine](https://user-images.githubusercontent.com/54680536/89717404-0ba6db00-d9d4-11ea-8619-77db7d248141.jpg)

Lastly We Used CRAFT Model for it's effictive ness with less data.

CAN BE IMPROVED MORE:
    with using both CRAFT AND EFFDET we can know which text belong to which box and we need not to other processing

#### Recognition of Text 
Our model demonstrate a higher accuracy with use of Transfer Learning of OCR-Tesseract LSTM on our annotated dataset and can be highly scalable to our documents with small amount of labeled training.

For this task aslo we applied two methods:
* Use Pytesseract [https://pypi.org/project/pytesseract/]
    * Advantage : We can extract text easily from text
    * Add other
* Use Text Recognition [https://arxiv.org/abs/1904.01906]
    * Advantage : It support Multilanguage and no need to optimize and we can train with our data
    * Distadvantage : It's performance little worse than Pytesseract.

Lastly We Used Pytesseract for it's effictiveness with this sample data.
