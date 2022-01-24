# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Seinfeld Script Generator
---
### Problem Statement

In 1996, Bill Gates wrote an essay whose title became so prevailing that every marketer knows and quotes-- "Content is King." In his essay, he believed that content "is where I expect much of the real money will be made on the Internet." Fast forward to today, his idea still stands the test of time, if not turns even truer.

We live in a world filled with content. Tehcnology has significantly lowered the bar of creating and consuming content. With the explosion of social media, the landscape of marketing has been changed completely. 10 years ago, it was fancy for a brand to have a Facebook page; now it's a must, along with a list of other must-haves: Instagram, Twitter, Pinterest, to name just a few.

To stay relevent, brands need to constantly supply content with high quality for informational or entertainment purpose, which sets a new requirment for marketers: not only do they need to generate content with great quality, they need to generate it more and fast, as well as with minimum costs possible. The good news, as Bill stated, is that "no company is too small to participate." However the question remains, how do we achieve all those? 

That is where text generation comes in. 

Recent years has witnessed series of jaw-droppping breakthroughs in the field of natural language processing, with new pre-trained NLP models producing state-of-arts results on various tasks from sentiment analysis to question answering. From a marketing's perspective, the applicatoin of text generation is endless: community engagement, developing derivatives based on an existing IP, A/B testing and so forth. Being able to utilize the text generation models will be a compelling advantage that sets the brand apart. 

This project aims to use text generation models to build a script generator for the popular 90s sitcom _Seinfeld_, with the hope to showcase marketing and operation professionals how far text generation can go and be applied with a vivid example. TV writers get paid \\$26,832 per 30-minute prime time episode. For a phenomenal show like _Seinfeld_, the price tag to hire a writer is much higher. In fact, one of the show's main writers is the leading actor Jerry Seinfeld himself, who made a staggering $13,000 per line arriving the final season of the show. Imagine that NBC would like to initiate a marketing campaign featuring reimagined _Seinfeld_ scenes, the script generator will help to afford the idea and likely achieve it faster.

Specific questions this project would like to answer:
1. Can a RNN-LSTM model be used to generate scripts with meaningful sentences in the same formality as the input data?
2. Does word level RNN-LSTM models outperfrom character level ones and why?
3. Is it possible to fine tune pre-trained GPT-2 model to generate _Seinfeld_-specific scripts? 
4. Can diversity be specified to meet different creative demands?
4. Which of the GPT-2 and RNN-LSTM perform better with regard to the form, content and speed of the script generation?
5. What are the thredholds of data size for each model?
6. Can these models be trained with a budget-friendly method?

### Contents

* [Data Aquisition and Cleaning](#data_aquisition_and_cleaning)
* [Exploratory Data Analysis](#exploratory_data_analysis)
* [Modeling and Tuning](#modeling_and_tuning)
* [Evaluation](#evaluation)
* [Script Generator API](#script_generator_api)
* [Conclustions](#conclusions)
* [Limitations and Next Steps](#limitations_and_next_steps)
* [Reflections](#reflections)
* [System Requirements](#system_requirements)
* [References](#references)

<a id='data_aquisition_and_cleaning'></a>
### Data Aquisition and Cleaning

The data used for this project is the scripts of all season episodes of _Seinfeld_. The only data source I found that allows web scraping is Internet Movie Script Database ([IMSDb](https://www.imsdb.com/)), which is a renowned resource for movie and TV scripts. 

As the source does not have an API for scraping, I built a function to automate the web scraping process. The function is composed of three parts: 
1. Locates the dedicated page for _Seinfeld_ with the list of all episodes; 
2. Locates the dedicated pages for each episode; 
3. Scrapes scripts on each episode page.

It was not expected the data was in fact not in good shape after being scraped, considering that it was a finished product and should have a standard format that is ready to go. However, upon discovery, massive formatting errors and redundant information were staggered throughout the data. 

The best efforts have been made to tidy up the data which include: 
1. Removed datapoints that have less than occurance because these data are not likely to provide any valuable inference for the model
2. Corrected typos, redundant character names
3. Corrected misplaced lines due to the wrong formatting of the data source
4. Dropped rows where dedicated to two speakers
5. Dropped duplicate scripts

The cleaned data before feature engineering has a shape of (44,661, 3) as compared to the original shape of (54,211, 3). Most of the reduction came from duplicated lines therefore I don't believe doing this will hurt the quality of the data. 

<a id='exploratory_data_analysis'></a>
### Exploratory Data Analysis

I engineered a couple of features for EDA purposes including ```word_count```, ```sentiment_score```, ```sentiment```("positive", "negative" or "neutral"), and ```line_no_par```(lines excluding the script description inside parentheses).

Overall the average word count per line is 13 words. The distribution of the word count per line is highly skewed to the right. The max number is 399, which was expected, as the show always start with Jerry Seinfeld doing a comedy stand-up which has long lines.

<table><tr>
<td> <img src="./charts/dist_word_counts.png" alt="Drawing" style="width: 375px;"/> </td>
</tr></table>

Below are the charts of top 15 characters with most lines and most line words. Not surprisingly, the four leading characters have significantly the most lines and words in the script. Jerry Seinfeld is the single character with most lines and words and led the charts by significant numbers, followed by George, Elaine and Kramer. I can also rougly concluded that the number of lines a character has is strongly correlated to the number of words spoken.

<table><tr>
<td> <img src="./charts/top15_most_lines.png" alt="Drawing" style="width: 500px;"/> </td>
</tr></table>
<table><tr>
<td> <img src="./charts/top15_most_words.png" alt="Drawing" style="width: 500px;"/> </td>
</tr></table>

The result of sentiment analysis surprised me. Despite the display of huge differences in characters on the screen, the lines of the 4 characters showed almost identical mean and distribution of positive, neutral and negative sentiments.  As indicated by the visualizations, all four of them have mostly neutral sentiment, followed by positive and then negative. However upon further thinking, the result might be due to the fact many _Seinfeld_ lines were sacarsm, which machine hasn't been trained well enough to tell.

<table><tr>
<td> <img src="./charts/top4_sentiment_percentage.png" alt="Drawing" style="width: 500px;"/> </td>
</tr></table>
<table><tr>
<td> <img src="./charts/top4_sentiment_describe.png" alt="Drawing" style="width: 450px;"/> </td>
</tr></table>

I also explored the most used words and n-grams by characters using default and custom stop words, from which I discoverd some patterns of the characters' line however didn't find much of distinctions among the characters. The characters tend to use a lot of interjection phrases such as "ha ha", "yeah", "oh", etc as well as negated expression such as "don know", "don want", "don think".

I created word clouds for each character for better visualizations. As shown below, the four characters have practically excatly the same most frequents words: "dont" "know" "like" "right".

<table><tr>
<td> <img src="./charts/jerry_wordcloud.png" alt="Drawing" style="width: 375px;"/> </td>
<td> <img src="./charts/george_wordcloud.png" alt="Drawing" style="width: 375px;"/> </td>
</tr></table>

<table><tr>
<td> <img src="./charts/elaine_wordcloud.png" alt="Drawing" style="width: 375px;"/> </td>
<td> <img src="./charts/kramer_wordcloud.png" alt="Drawing" style="width: 375px;"/> </td>
</tr></table>

Below is a brief summary of most used words by four characters collectively and indiviually. The very left column shows the frequent words/phrases shared by all four characters. I labeled the wordes/phrases into four categories: interjection words, negated expression, character names, and meainful one(s). Upon examining, my takeaway is -- it is truly a show about nothing.

|**Shared**|*Jerry*|*George*|*Elaine*|*Kramer*|
|---|---|---|---|---|
|**Interjectionr**:`yeah, hey, oh,  oh yeah, hey hey, oh hi, yeah yeah, yeah right, yada yada yada`|None|`ha ha`|`ha ha, uh uh`|`uh uh, ho ho`|
|**Negation**:`don know, don want, don think`|`don understand, don care`|`don understand, don like`|`don understand`|`don like`|
|**Character Names**:`Jerry, George, Elaine, Kreamer`|`Uncle Leo`|None|`Mr. Peterman, Mr. Pitt`|`Newman`|
|**Meaningful**:`get out`|`coffee shop, wait second`|`coffee shop, wait minute`|`coffee shop, wait minute`|`coffee table`|

Also it was during this process that I realized there are still some formatting errors in the data that was not possible to efficiently clean out within the time limit. Considering the amount is not too huge, I decided to work with what I had.

Upon conducting EDA, I came up with some conslusions and hypotheses:

1. The original data is a lot messier than it was expected. After performed data cleaning to my best efforts, I know there are still some noises within the data that are nearly impossible to be cleaned up entirely. Therefore I should expect some errors in my generated scripts.
2. Jerry, George, Elaine and Kramer are definite leading charaters with most lines and line words and therefore I assume that most of the model outputs will consist of dialogues among them.
3. As the average line length is 13 words, I may not want my model to learn the scripts with a number smaller than that, otherwise not enough information will be fed into the model to effectively learn. In the similar sense, considering the nature of conversations, I may not want my model to have too many lines to look at either, as that way will create noises.
4. Although I perceived distinctive personalities for the 4 leading characters from the show, I didn't see enough distinction from their lines. They have similar sentiment results and simlar word usages. If anything, Kramer may be the only one who shows slight difference from his three friends.
5. I would expect to see many repetitions of "meaningless" high frequency words such as "yeah yeah", "oh yeah", "don know", "don think", "don want" in the generated scripts.

<a id='modeling_and_tuning'></a>
### Modeling and Tuning

Three types of models were ran, in the spirit to find pros and cons of each model and provide more options for the audience based on their needs. 

The first model was a character-level RNN/LSTM model. The model was trained on Kaggle's GPU and used only the first 500,000 characters (1/7 of the total data) to ensure the running of GPU. The sequence length was set at 100, as a conservative attempt to test out the model. The architecture used three LSTM layers as well as two dense neural network layers.

|**RNN Architecture**|*Neurons*|Dropout|Parameters|*Output Shape*|
|---|---|---|---|---|
|**Embedding Layer**|None|None|5900|*( None, 100, 100 )*|
|**LSTM 01**|512|0.1|1,255,424|*(  None, 100, 512 )*|
|**LSTM 02**|512|0.1|2,099,200|*( None, 100, 512 )*|
|**LSTM 03**|512|0.1|2,099,200|*(  None, 512 )*|
|**Dense**|256|None|131,328|*(  None, 256 )*|
|**Dense**|128|None|32,896|*(  None, 128 )*|
|**Dense Output**|18,725|None|7,611|*(  None, 59 )*|

With a total training time of 2 hours with 30 epochs, I got a loss of 1.0687. The generated scripts from the model perform greatly with the learning the format and certain spellings; however, even with the relative complex model set up, all scripts generated contain heavy repetition from the characters most frequently used words.

With the belief that more data should be used as well as use word as a token relieve model from learning spellings, I did the second RNN/LSTM model based on words. I set the sequence length to be 40 words which correspond to approximately 3-4 lines, which is an increase from the character level model.

|**RNN Architecture**|*Neurons*|Dropout|Parameters|*Output Shape*|
|---|---|---|---|---|
|**Embedding Layer**|None|None|400,000|*( None, 40, 40 )*|
|**LSTM 01**|512|0.1|1,132,544|*(  None, 40, 512 )*|
|**LSTM 02**|512|0.1|2,099,200|*( None, 512 )*|
|**Dense Output**|18,725|None|9,605,925|*(  None, 18725 )*|

A batch generator function was built to break through the hardware limitation, a diversity function was added at the script generation stage so that customization could be added. The final script generation function takes in ```seed text```, ```length```, and ```temperation```(the diversity level).

The word-based RNN/LSTM ran for 6.5h in total with thirteenth epochs has the best loss of 3.13. And the generated results saw significant improvements across temperature values. The model learned to follow the general pattern of the original script, which is the speaking ```character first: ``` and the corresponding line. However as part of the preprocessing, sepcial characters were separated from the words for tokenization purposes, which made the model unable to learn about most punctuation rules. The special characters were added back to the text using a manual function. In addition, The sentences and names do not capitalize, which is certainly an area to work. 

Last but not least, I utilized transfer learning to fine tune a GPT-2 model using _Seinfeld_ data. The GPT-2 is a transformer-based model which uses attention mechanisms to predict the next word. It has 12 attention layers with each has 12 individual heads. GPT-2 has been proven to achieve state-of-art results in text generation and was considered the most powerful language model. 

Thanks to the wrapper created by [Max Woolf](https://github.com/minimaxir/gpt-2-simple), I was able to train the model fairly easily. The model allows various level of fine tuning, including ```prefix```, ```length```, ```temperature```, etc. The smallest GPT-2 size was used for this project for a balance of speed and effiency.

<table><tr>
<td><img src='./img/gpt_2_architecture.png' alt="Drawing" style="width: 200px;"/></td>         
</tr></table>

After training on Google Colab GPU for about 2 hours, I achieved the loss of 0.83. 

The GPT-2 model has generated consistant high quality results. A few generated texts are very human-like which consists of simple logic arguments. The model certainly learned the style and content of the data with rare errors as stated. And thanks to the attention model, the generate text has exciting results in learning about context within the generated text. Also due to GPT-2's huge number parameters, external information often occurs in the generated text. GPT-2 didn't perform as well as word-level RNN/LSTM model in diversity. If the diversity level is too high, the scripts tend go off to its own track, while even when the temperature is not too low, the scripts 


<a id='evaluation'></a>
### Evaluation
For text generation models, loss seems to be so far the only quantitative metrics used to evaluate the model. However, for my project the character-level RNN-LSTM model has a lower loss than the word-level model, yet the word-level model performed significantly better. Therefore, the evaluation of model was done by me reading through every generated scripts. 

Upon reading hundreds of scripts generated by my models, I believe word-base RNN-LSTM model and GPT-2 model showcase their own pros and cons and therefore will use these two models to make suggestions for the audience of this project.

Among these two models, from the cost perspective, RNN model took 3.6 hours to train and 12 seconds to generate a 200-word script, while GPT-2 took 2 hours to train yet 40 seconds to generate the script of same length. The RNN model was fine tuned and trained on Google Cloud multiple times across the time span of 2 weeks which end up costing fewer than \\$150 while GPT-2 cost almost nothing. 

<a id='script_generator_api'></a>
### Script Generator API

An API for each model was developed using streamlit.io as an user-frinedly interface for non-technical users to try out the model. Below is a screenshot of the GPT-2 model app. 

<table><tr>
<td><img src='./img/api_gpt_screenshot.png' alt="Drawing" style="width: 500px;"/></td>         
</tr></table>

Due to the large size of GPT-2 model and the special tensorflow version requirment, so far the project was not able to deploy the APIs online. However, they can be ran locally. Please refer to system requirment section for details.

<a id='Conclusions'></a>
### Conclusions

Based on the models and results, following conclusions were drawn for the marketing professionals who are looking to utilize the models:
1. For text generation models, data is the key. If the dataset is smaller than 1MB, the GPT-2 model should be used, as the RNN would not effectively learn. Between the size of 1-6MB, either RNN or GPT-2 would provide good results, other considerations should prevail in deciding which model to go with. If the dataset is greater than 6MB, which is considered very large for text data, RNN should perform well.
2. The purpose of text generation is also critical when it comes to model choosing. RNN results are more loyal to the original vocabulary while GPT-2 often adds in external information that it learned from its existing paramters. If a marketing professional is looking to establish close bond with a fan community, perhaps RNN will fit better. For inspiration purposes, on the other hand, GPT-2 could provide many inspiration. Althought both models exceed the boundary every once in a generation. GPT-2 generates almost ready-to-go text that are close to perfect in terms of format and grammar however RNN works amazing generating abstract, comedic, and casual texts.
3. From the time and cost perspective, both models have their pros and cons. If a marketing professional would like to generate chunks of texts, then RNN is the way to go due to its lightweight.
4. As RNN is a model built from scratch, additional fine tuning is possible to perfect it, meanwhile GPT-2, due to its pre-trained nature, is not likely to be change significantly.

<a id='limitations_and_next_steps'></a>
### Limitations and Next Steps

Limitations of this project include: 1. as the data was a lot messier than expected, it added noises to the modeling which can be noticed from some of the generated results; 2. The word-based RNN-LSTM model had a hard time reducing its loss function; 3. Additional time/efforts on fine-tuning RNN model needed; 4. Both models are highly dependant on cloud computing which poses potential technical challengese; 5. No effective evaluation metrics have been utilized yet. So far the evaluation is all manual which slow and objective; 6. GPT-2 as an
internet-trained models has acknowledgably internet-scale biases; 7. Unable to deploy API due to dependency version conflicts.

For next steps, I would like to: 1. incorporate attention mechanism to RNN-LSTM model; 2. try training the existing data on some joke dataset to see if it could make the script funnier; 3. create a function to printouts the model's learning progress throughout the training; 4. build an automated way to evaluate the generated texts; 4. improve the API to be more user-friendly.

<a id='system_requirements'></a>
### System Requirements
A virtual environment of Python 3.6 was suggested to run the GPT-2 model and streamlit API as the GPT-2 requires a tensorflow with 1.1*.

In your virtual environment where ```app.py``` is located, run the following code to initiate the API:

`streamlit run app.py`
`streamlit run rnn_app.py`

Also note that recent updates of python may result in a failure to load ```.hdf5``` and ```.h5``` files.

<a id='references'></a>
### References

* [`Understanding LSTM Networks`] (GitHub): ([*source*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/))
* [`“Content is King” — Essay by Bill Gates 1996`] (Medium): ([*source*](https://medium.com/@HeathEvans/content-is-king-essay-by-bill-gates-1996-df74552f80d9#:~:text=Ever%20wondered%20where%20the%20phrase,as%20it%20was%20in%20broadcasting))
* [`OpenAI GPT-2: Understanding Language Generation through Visualization`] (Towards Data Science): ([*source*](https://towardsdatascience.com/openai-gpt-2-understanding-language-generation-through-visualization-8252f683b2f8))
* [`Examining the Transformer Architecture`] (Towards Data Science): ([*source*](https://towardsdatascience.com/examining-the-transformer-architecture-part-1-the-openai-gpt-2-controversy-feceda4363bb))
* [`Examining the Transformer Architecture`] (Towards Data Science): ([*source*](https://towardsdatascience.com/examining-the-transformer-architecture-part-1-the-openai-gpt-2-controversy-feceda4363bb))
* [`The unreasonable effectiveness of Character-level Language Models`] (Jupyter): ([*source*](https://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139))
* [`The unreasonable effectiveness of Character-level Language Models`] (Jupyter): ([*source*](https://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139))
* [`Generating TV Script using LSTM Network | Keras`] (Medium): ([*source*](https://medium.com/coloredfeather/generating-a-tv-script-using-recurrent-neural-networks-dd0a645e97e7))
* [`The Illustrated Transformer`] (Github): ([*source*](http://jalammar.github.io/illustrated-transformer/))
* [`How Biased is GPT-3?`] (Github): ([*source*](https://medium.com/fair-bytes/how-biased-is-gpt-3-5b2b91f1177))
* [`Improving Language Understanding by Generative Pre-Training`] (Amazon AWS): ([*source*](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf))
* [`Alternative Structures for Character-Level RNNs`] (arXiv): ([*source*](https://arxiv.org/pdf/1511.06303.pdf))
* [`Attention Is All You Need`] (arXiv): ([*source*](https://arxiv.org/pdf/1706.03762.pdf))
