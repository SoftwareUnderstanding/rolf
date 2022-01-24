### NLP_text_summarization_apps

There are two apps: nlpapp.py and nlpapp2.py

Both APPs allow the user to input text and have a model or models summarize the text.

* nlpapp.py:  uses a neural net transformer model to read text and generate a summary of the text passed to the app. The App uses a Large BART model from the Huggingface transformer library.The Model uploads once only when app is first run. After uploading you can test the pre-trained model on text as much as is needed. 
 

* nlpapp2.py: offers the user the BART model and also the LexRank model to summarize text. LexRank is a count based model. It is relatively faster than BART on larger passages of text but personal preference is to work with BART as it produces more insightful/concise summaries on larger passages of text.

Before running the app from the commandline check with the requirements.txt file which python libraries are used in both apps. If needed , pip install the requirements.txt ahead of running the apps.

To run either app locally enter in your commandline:  streamlit run nlpapp.py or streamlit run nlpapp2.py

Citations:
BART research paper: https://arxiv.org/abs/1910.13461

LexRank research paper: https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html

Aug/2021

