# Capstone Project
## Agent Assistant - enabling agent efficiency using ML
#### Shashank Rajadhyaksha
#### *DSIR22221-E Capstone Project, Presented 5.14.21*

## Executive Summary

Agents in various operational sites have to do a vast variety of tasks and have to multitask using the tools they have.  These agents are the face of the company to a lot of customers and the customer experience provided by these agents is very important in driving customer's perception of the product and the company.

In global companies, some agents also have to service accounts of accountholders who speak other languages. So it is important that agents have multiple tools at their fingertips that enable them to do their jobs better.  In addition, better tools can drive more efficiency which can help to reduce staffing demand and drive lower costs for organizations.

This project was to develop a suite of tools (called AA for Agent Assistant) to help agents in their efforts in servicing customers.  There were 3 ML based tools that have been developed here to assist the agents in their effort:

#1 **English to Spanish translator**:  The intent of this tool is to enable agents to communicate with Spanish-speaking customers. This was built using English and Spanish datasets of multiple sentences.  It was built using natural language processing tools in [scikit-learn](https://scikit-learn.org/stable/) and [NLTK](https://www.nltk.org/).  A sequence-to-sequence LSTM model was built using the underlying data to develop a model to enable translation.  The model's accuracy was around 80% for the test and train population.

#2 **Sentence Completion**: The intent of this tool is to enable an agent to complete their sentences after they had typed the first few words making them faster at their job.  This would be akin to how gmail completes sentences after you have typed a few words.  The same English sentences that were used for translation were used.  However, the input feature was the first few words from the English lines and the last few words from the same English line was the target for the model.  Similar to the translation algorithm, another sequence-to-sequence LSTM model was built to enable completion.  The model's completion accuracy is displayed to be ~80% for the test population, however, the proposed sentence completers are not as clear. 

#3 **Chatbot to help agents**:  The intent of this tool is to enable an agent to type in a question about some credit card feature and it would then answer the question.  Using the FAQ from a bank's webpage and randomizing permutations, a chat corpus was created.  This is then used to used to train the model where a question is mapped to the intent behind the question.  Intent can range from how to apply, how to earn rewards, what is reported to the credit bureaus etc.  Each intent has a unique answer and the chatbot would respond with that answer.  The model accuracy here was over 95% primarily because the chat corpus was built around the FAQ words.

For the Translator and Completer, the execution of training and inferencing the LSTM model had to be developed separately. Given the long run-times, additional functions were developed to continue with the epoch training after a few epoch runs.  Also a set of decoding functions had to be developed to read in the text and load that into a certain form for the model's encoder and decoder to use and translate/complete.  A streamlit app was developed for all the 3 models so that AA can be demonstrated to agents and interested organizations.

Some challenges that were faced (especially for the Translator and Completer), was that not all lines could be used due to system & memory limitations in the training process.  This limited the training of the model and the translations/completions it can come up with.  Also the model wouldn't run on a laptop and had to be run on Google Cloud Platform in order to run over a few hours.  Both Translator and Completer needed over 50 epochs which took over 5 hours to run on Google Cloud Platform.

Looking forward, the opportunity is to improve the algorithms and use more data in conjunction to improve the quality of the translation and completions.  For the chat model, the opportunity is to train the model on actual chat conversations so that it can be trained better on unseen data and develop better response.

 
## Problem Statement

Chat agents have to perform multiple tasks simultaneously that can slow them down.   They have to cover multiple languages at times, they have look up product questions so that they can get back to the customer quickly - and all these activities can slow then down as well as introduce the likelihood of errors.

**We aim to help chat operations agents** by developing tools using ML models to improve their efficiency and effectiveness.

We have used ML models to **have a translater that can translate from English to Spanish**, **a sentence completer that can finish their sentences after they have typed a few words**, and **a chatbot that can answer their product related questions**.  

---

## Datasets

### Data Collection
This project used the data for translater and sentence completer from the following source: 

Link to dataset of Spanish-English sentences: http://www.manythings.org/anki/spa-eng.zip

The initial dataset had ~128k rows with each row having the English and corresponding Spanish line.  This had been curated by lines coming from different sources.  There were about 19k duplicate English lines that were then de-duped from the dataset.  There were also sentences with very few or very high characters as well as sentences with fewer words.  Sentences with less than 18 characters and with less than 4 words and with more than 50 characters were removed.  The lower end counts were removed to not have very sparse matrices in the eventual dataset.  The above suppressions resulted in ~80k records for the Translator and ~50k records for the Completer.

For the chatbot, the data was sourced from a primary banks' website from the FAQ page with questions and answers.  The chat corpus was developed using words from the questions and permutations created to expand to multiple questions.  Each question was tagged to an 'intent' and the intent was mapped to an 'answer'.  For example, questions related to application of a card was tagged as CardApplication intent with a defined answer for that intent.


### Data Dictionary
Here are sample rows from the dataset that were used for the Translator, Sentence Completer and the Chatbot.

#### English - Spanish Translator
|Spanish Sentence	|English Sentence	|
|-	|-	|
|un perro tiene cuatro patas	|	a dog has four legs	|

#### English Sentence Completer
|English Pre	|English Post	|
|-	|-	|
|a dog has 	|	four legs	|

#### Chat Dataset
|question	|intent	|
|-	|-	|
|What information does Banco Uno® require when I apply for a credit card? 	|	Cardapply
|How can I report a lost or stolen credit card?? 	|	LostStolenCard
---


---

## Model Build for Translator and Completer
### Data Cleaning & Pre-processing Steps
- Lemmatized, tokenized and joined words to re-form sentences
- Standard processing such as converting to lower case characters, removing punctuations, removing extra spaces and digits for English & Spanish sentences
- Examined the data and removed duplicate English sentences

   
### EDA/Processing for Translator & Completer
- Created 2 new variables around count of characters and words for English & Spanish sentences and reviewed distributions by count of characters and words
- For Translator, kept lines with characters between 18 and 50
- For Completer, same pruning for characters and also deleted lines with less than 4 words

### Modeling & Evaluation of LSTM models
Translator and Sentence completers need NLP based models.  These models need to store the words and their sequence that comes together to form a sentence.  

This makes sequence-2-sequence LSTM models the ideal model to use since it has to retain memory of the past sequence to create the new words.  There are other options like using embedding and teacher-forced training that can further enhance the model.

This modelling exercise was influenced by Francois Chollet's usage of sequence-2-sequence LSTM models.  https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html.  Google's translation (https://arxiv.org/abs/1609.08144) and Facebook's translation were also based on LSTM albeit with a lot more encoder and decoder layers, more evolved algorithms and lot of memory / computation power to build these out.

Train and Test split were created and the model was trained through multiple epochs.  The local machine was not able to handle the higher memory resulting in the program crashing or running very slow.  So training was moved to the Google Cloud Platform with a Cuda instance, 8-CPU machines, 64GB RAM and 100GB disk. This made the epochs run for more lines and much faster. 

The training program was modified to store the model every 5 epochs.  In addition, another function was written to load the interim model as well as load other data required for training.  This needed converting the interim dictionaries that were created into pickled files that would then have to be read again for the model to continue learning using the loaded model.

In addition an inference model had to be developed, where one would pass the text that had to be translated as input, it would then predict the translation or sentence completion by reading the encoder and decoder configurations from the saved model - and it would then write the translated Spanish text or the completed sentence.

Here are the tactical steps for the model building process mapped to the scripts in the github:

- Translation from English to Spanish:
    - St1: reads in the text file which has Spanish/English sentences and cleanses data around removing punctuations, speciail characters, numbers etc.  It then removes duplicate English sentences and prunes short and long sentences and then creates the 'EngSpa.csv' file. 
    - St2: reads in the 'EngSpa.csv' file from the previous step, parses the 'eng' and 'spa' columns from the file to a dataframe, builds out lists to be used for modelling, creates dictionaries of words from input and target features with a forward and reverse view.  It then creates two 3-dimensional array (#lines * #words in the line * #tokens for input or target) that is then used for modelling and models are created using multiple epochs.

- Sentence Completer
    - St1:  reads in the 'spa.txt' file which has Spanish/English sentences and cleanses data around removing punctuations, speciail characters, numbers etc.  It then removes duplicate English sentences and prunes short and long sentences and then creates 'eng.csv' file. 
    - St2:  reads in the 'eng.csv' file from the previous step, parses the 'eng_pre' and 'eng_post' columns from the file to a dataframe, builds out lists to be used for modelling, creates dictionaries of words from input and target features with a forward and reverse view.  It then creates two 3-dimensional array (#lines * #words in the line * #tokens for input or target) that is then used for modelling and models are created using multiple epochs.
    - St3: reads in the model as well as dictionaries and other inputs for the model.  This then creates the function that can be used for completing sentences.

## Chat Model Build
### Data Cleaning Steps
- Removed special characters from the questions columns
- Lemmatized, Tokenized, cleansed and added words together
- Developed a column with permutation of different words from the questions column

### Preprocessing for NLP Model
- Tokenized data to remove punctuation 
- Lemmatized data so only singular forms of words remained
- Removed English stopwords

### Modeling for Chat
- Dataset included ~36k rows of chat related questions and the corresponding intent
- There were about 70 different intents (or classes)
- X-variable: various questions generated from the FAQ list of questions, y-variable: 70 different classes (Examples are Card Application, Adding Authorized user, reporting lost stolen cards etc)
- Vectorized data using Tf-IDF Vectorizer with 2 grams and max_df=2.0, min_df=5
- Split data into train and test sets using train-test-split with split ratio of 
- Created a multiclassification model using basic Logistic Regression 
    - Results:
        - Train score:  .997
        - Test score: .996
    - Interpretation:
        - The logistic regression model scored extremely well.  This was mainly because the 'chat corpus' was developed using permutations of the questions from the FAQ.  So there were not many unseen words - which would happen in an actual chat conversation.  A next step here would be to train the model on actual chat questions and then map to the intent and evaluate how the model would do.
        - Tested a Neural Net model but the performance was so strong from the Logistic Regression model that there was not a lot of value in evaluating different models.  This would be more relevant when one is building the model on actual varied chat transcripts.

Here are the tactical steps for the model building process mapped to the scripts in the github:
- St1: used to create a chat corpus from downloading the FAQ dataset.  
- St2: reads in the chat corpus, does standard data processing and then models logistic regression.

## Agent Assistant App using Streamlit

We were able to achieve our goal of helping Chat agents by providing them tools to enable translation, sentence completion as well as a chatbot to quickly look up product terms.  

The app has 3 functionalities that leverages models to make predictions:
- Translates sentences typed in English into Spanish
- Completes a sentence after a few words have been typed
- Provides credit card product related answers after a user types in their question

We believe that this app can improve efficiency for chat agents by providing them quick answers at their finger-tips during various interactions.  

The streamlit code is stored in main folder and is called GlobalStreamlit.py.  It creates a side-frame which allows to pick between the 3 apps and then the app can be used to try out one of the 3 options.

## Conclusions & Future Directions

The 3 different models have varying levels of success and have different future directions. 

The translator does an okay job on words it has been trained upon.  It translates short sentences well.  However, it falls short when it sees an unknown word or is given longer sentences.  This is mainly driven because the model was trained only on 18k lines due to limitation of how much space was needed for LSTM models.  The next step here is to use enhanced algorithms which have embedding features, more layers using more computing power & memory (so it can consume more lines) to develop better translations.

The sentence completer wasn't as good as expected.  Similar to the translator, the completer was trained only on 40k rows due to memory limitations.  The completions were somewhat limited - potentially because of so many combinations being possible for sentence completions.  The next step here would be similar that we would need to enhance algorithms that use memory better and use more data to build better completers.

The chat model does an inherently solid job of figuring out intent if the right words are in the question text.  It then accurately figures out intent and plays back the answer.  In order to make the model more successful, one should train the model on actual chat transcripts (these were not available for bank or similar institutions) with the correct intent labelled.  This would significantly improve the ability to comprehend different questions and come up with the right answers.

Accompanying presentation available at https://docs.google.com/presentation/d/1wUOmMMDLDxrQoH26EReC_HTfhWB8iuJyXBeBmt7EOsE/edit#slide=id.gd62292d7e6_0_37

---

### File Structure

```
Capstone
|__ COF_CS_Chat
|   |__ data
|      |__ChatCorpus.csv
|      |__IntentAnswers.csv
|      |__QuestionSet.csv
|   |__ models
|       |__cs_model.p ##Created from Logistic Regression model
|   |__ 
|       |__St1_CreateChatCorpus.ipynb #Creates ChatCorpus from the QuestionSet.csv
|       |__St2_NLP_model_text.ipynb #Creates the NLP model (cs_model.p) that is used for Streamlit
|__ SentenceCompletion
|   |__ data
|      |__eng.csv #English CSV file created after St1 below is run
|      |__ifd.p #Pickle files below for dictionaries of words
|      |__rifd.p
|      |__rtfd.p
|      |__tfd.p
|   |__ models
|       |__training_model_gcp.h5 ##Created from St2 below
|   |__ 
|       |__St1_Read_EDA_Eng.ipynb #Reads the spa.txt file, does basic EDA
|       |__St2_DataProcessModel_Eng.ipynb #Creates the model (training_model_gcp.p) for predictions
|       |__St3_CompleteSentence.ipynb #Inference model to do the sentence completions
|       |__spa.txt #Raw data file that is used to then create the eng.csv file
|__ TranslateEngSpan
|   |__ data
|      |__EngSpa.csv #Spanish English CSV file created after St1 below is run
|      |__ifd.p #Pickle files below for dictionaries of words
|      |__rifd.p
|      |__rtfd.p
|      |__tfd.p
|   |__ models
|       |__training_model_gcp.h5 ##Created from St2 below
|   |__ 
|       |__St1_Read_EDA_Eng.ipynb #Reads the spa.txt file and does some basic EDA
|       |__St2_DataProcessModel_Eng.ipynb #Creates the model (training_model_gcp.p) for predictions
|       |__St3_TranslateNewText.ipynb #Inference model to do translations
|       |__spa.txt #Raw data file that is used to then create the eng.csv file
|__ README.md
|__ GlobalStreamlit.py
|__ ML_AI for CS agents.pdf
|__ requirements.txt
```