# Language-Classification-Using-Naive-Bayes-Algorithm
[![HitCount](http://hits.dwyl.com/PaulSudarshan/Language-Classification-Using-Naive-Bayes-Algorithm.svg)](http://hits.dwyl.com/PaulSudarshan/Language-Classification-Using-Naive-Bayes-Algorithm)

The project on Language Classification using Naive-Bayes algorithm deals with classifying and identifying the language of the input string into its correct language category. In order to demonstrate the use-case the model has been trained to detect three different languages. The languages chosen for this model are Slovak (sk), Czec (cs) and English (en). The purpose for choosing slovak and czec is that both of these languages are very similar in the way they are spoken, so if the model is able to distinguish between these two given languages that will ensure the robustness of our model to classify between other dissimilar languages with a very good accuracy.
![](images/diff_lang.jpg)
# Application of Project:
In order to understand the application of this project with a concrete example we can consider it as an initial step before performing Language Translation. If the user is not aware of the language he wants to translate using a Language Translator, we can perform a language classification of the input string and then apply appropriate Language Translator tool language conversion. For example Google Auto Detection.

# Dataset Description
## [Slovak Wikipedia Entry](https://sk.wikipedia.org/wiki/Jazveč%C3%ADk)
Mnohí ľudia, ktorí vidia na ulici jazvečíka s podlhovastým telom vôbec nevedia o tom, že tento malý štvornohý a veľmi obľúbený spoločník je pri dobrom výcviku obratným, vynikajúcim a spoľahlivým poľovným psom. Ako poľovný pes je mnohostranne využiteľný, okrem iného ako durič na brlohárenie. Králičí jazvečík sa dokáže obratne pohybovať v králičej nore. S inými psami a deťmi si nie vždy rozumie.

## [Czech Wikipedia Entry](https://cs.wikipedia.org/wiki/Jezevč%C3%ADk)
Úplně první zmínky o psech podobných dnešním jezevčíkům nacházíme až ve Starém Egyptě, kde jsou vyobrazeni na soškách a rytinách krátkonozí psi s dlouhým hřbetem a krátkou srstí. Jednalo se ale o neustálený typ bez ustáleného jména. Další zmínky o jezevčících nacházíme až ve 14 - 15. století. Jedná se o psa, který se nejvíce podobá dnešnímu typu hladkosrstého standardního jezevčíka.


## [English Wikipedia Entry](https://en.wikipedia.org/wiki/Dachshund)
While classified in the hound group or scent hound group in the United States and Great Britain, the breed has its own group in the countries which belong to the Fédération Cynologique Internationale (World Canine Federation). Many dachshunds, especially the wire-haired subtype, may exhibit behavior and appearance that are similar to that of the terrier group of dogs.

# Tools and Libraries Used :
1. Jupyter Notebook
2. Pandas Library (for Data Manipulation)
3. sklearn
4. joblib (for pipelining)
5. pickle (for converting existing python object into character stream)

# Methodology :
1. We import the specific language data from the above mentioned sources and store them in the form of a dictionary as key value pairs. The 'key' being the Language Name and 'value'being the sample of the respective language.
2. We derive some statistics from the data like the number of sentences present in a particular language sample, total number of words, number of unique words etc.
3. After gaining some insights about the data we perform data cleaning and pre-processing by removing punctuations, digits, unnecessary symbols and store them back as key value pairs in refined form.
4. After pre-processing we split the data into dependent (language name) and independent (language sample) variables as x_train and y_train.
5. Next step is to perform the vectorisation of the x_train(independent variable) by using a library called CountVectorizer (from sklearn.feature_extraction.text import CountVectorizer), this converts our language sample into a sparse matrix of 0s and 1s.
6. We perform the exact same steps for out validaton dataset.
7. The next step is to initialise the model parameters and training the model with our training data that we just created. The model that we will be using for our purpose is Multinomial Naive-Bayes Algorithm. Naive Bayes model is best chosen for this task because Naive Bayes classifiers work by correlating the use of tokens (typically words, or sometimes other things), with different sets of languages and then using Bayes' theorem to calculate a probability that a particular string of language belongs to which category of languages.
8. Certain parameters in the model training can be set to obtain better results one such is 'alpha' value which is also called 'smoothening' constant which takes care of the words which are not present in the language sample. 
9. One other parameter is fit_prior which is set to False to avoid waitage of prior_probability while classifying, to assume conditional independence across all features.
10. Joblib library has been used to create a pipeline of out model (Naive-Bayes) and the CountVectorizer that we used during the pre-processing.
10. Lastly, to increase the model performance we have used a technique of using subwords which has been derived from the paper mentioned here https://arxiv.org/abs/1508.07909 , using this technique of preprocessing the sample into various frequently occurring subwords, the accuracy of model was significantly improved.

# Results
### Let's Look at the interesting results we have obtained :

![](images/english.png)
![](images/slovak.png)
![](images/czec.png)



# Conclusion
Thus, a model to classify three different languages (English, Slovak, Czec) have been trained and their outputs have been verified for model accuracy.
