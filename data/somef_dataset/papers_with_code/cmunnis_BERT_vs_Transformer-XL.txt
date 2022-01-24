# BERT_vs_Transformer-XL

## A Comparison of Two NLP Frameworks for General Research Purposes

The goal of Natural Language Processing (NLP) is to train computers to analyze human language. The widest-used versions of NLP are used in spell-check and grammar-check programs, but more advanced versions have been developed into tools used for much more than just identifying context within search queries. NLP is becoming increasingly more useful for researchers to summarize large amounts of data or long-form documents without the need for human supervision. Our project will examine two powerful NLP algorithms, BERT and Transformer-XL, in their abilities to extract and summarize data from chosen pieces of literature. Both have the attention model Transformer as their base. “[Transformer-XL] consists of a segment-level recurrence mechanism and a novel positional encoding scheme” (Dai, et al. 2019), meaning it takes segments of data and not only individually analyzes each segment, but also references segments against each other for increased accuracy regarding context. BERT focuses on working around the Transformer constraint of unidirectionality, where context is analyzed in only 1 direction, leaving room for error when the context from the other direction is needed. The strategy for its bidirectionality is “using a ‘masked language model’ (MLM) pre-training objective,” which “randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word using only the [left and right] context” (Devlin, et al. 2019). We will provide each algorithm with the same dataset and judge the results for each algorithm on its accuracy compared to its execution time.

#### Works Cited

Dai, Zihang, et al. “Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.” ArXiv.org, Cornell University, 2 June 2019, arxiv.org/abs/1901.02860.

Devlin, Jacob, et al. “BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding.” ArXiv.org, Cornell University, 24 May 2019, arxiv.org/abs/1810.04805.


## Weekly Progress Reports

**2/14/20**:
- We've uploaded our abstract to the MassURC website and specified our needs for presentation. We created the GitHub repository for this project.

**2/22/20**:
- We updated the abstract to have quotes from the two research papers we're going to reference in our project. Ed has been working on setting up his company's server to run BERT, and will be providing access to Vincent and Connor soon. Vincent and Connor have been researching how to use/understand the results from both algorithms.

**2/28/20**: 
- It was a bit close, but we have successfully been able to implement BERT onto our server, and are able to demonstrate it executing and working. With the structure of BERT implemented, our goal now shifts from the basics of BERT to changing its dataset manually.

**3/6/20**: 
- Alongside the change in data set, we are also in the process of modifying the function to output, and save the generated results, the input, and the amount of time, in milliseconds into a text file with a similar syntax to that of a JSON file. Both this, and changing our bert's data set (from the IMDB Database to the Wikipedia Database), are still a work in progress, but we are getting closer to its completion.
   + We've converted our TF BERT program to operate on Tensorflow 2, improving functionality and allowing use with Keras. 
   + We've installed transformer-xl onto our server and are writing a keras script for building, finetuning and testing our transformer-xl model. 
   
**4/2/20**: 
- *Overview*: 
   + Amongst other goals, scripts are being developed to significantly speed-up the testing and comparing process, to hopefully increase development efficiency.
- *Edward*:   
   + Implemented entity-recognition.py for the Bert model, which can be viewed within our repository. This script is the counterpart to our entity recognition script for transformer xl. Both scripts read through a prepared script and identify entities that are containned within the scripts. We can use the combination of accuracy of the two models ability to find entities as well as the specificity of the entity categorization to judge the efficiency of the two models against each other. Our testing process will allow us to see how each model responds as the text used for entity analysis grows longer. 
   + Additionally, background research will need to be done to see if bias is introduced due to the entity-recognition script utilizing pytorch rather than tensorflow-2, to see if it would be valuable to refactor the pipeline transformer-xl function to utilize pytorch as well.    
- *Vincent*:   
   + Currently in the process of creating intuitive testing and comparing scripts that will allow for increased productivity, as well as reformat the output of both BERT and Transformer-XL in a way that will allow both to be comparible to each other, and their output sent to different columns of a spreadsheet file, along with the base input, for ease of recording and comparing.
- *Connor*:    
   + Continued to: read up on Machine Learning; learn Python to be able to help make the necessary changes; and look up examples of how to find/understand output for each model.
   
**4/12/20**:   
- *Edward*:    
   + Implemented BERT question answering, need to increase data pool for question answering to more than 512 tokens, or build a token delivery system instead. Will replicate BERT question answering into transformer xl. 
- *Vincent*:   
   + Comparable data sourced from bert is currently functionally saved. However, further work is needed to encapsulate all of the required information, as well as identical formatting
- *Connor*:    
   + Worked with Vincent on having the BERT and Transformer-XL outputs made readable and itemized.

- **Goal for Next Week**:
   + We need to implement spreadsheet input functionality, make a unified spreadsheet input and output function with an increased data pool size past 512 tokens, and implement question answering on Transformer-XL to mirror the question answering on BERT.

**4/18/20**:
- *Edward*:
  + Implemented an improved BERT QA system that can be found within bert-qa-advanced.py. This file allows us to run large batch tests on BERT's question answering, it also has an improved output style.
  + Implemented a basic QA pipeline for transformers-xl, also reviewed documentation on how others have utilized transformer-xl for QA: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15766157.pdf, as this type of behavior isn't what transformer-xl is designed/optimized for. 
  + Fixed our transformer-summarization model to not use BERT tokenization, read this source material to see how text summarization could be improved with transformer-xl https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15776950.pdf
   + For next week I will be changing our other tests (Named Entity Recognition, Next Sentence Prediction, Token Recognition and Summarization) to utilize the same testing structure. 
- *Vincent*:
   + Generated a series of edge-case a logical-understanding cases for the Q&A testing.
   + Also worked on creating a more dynamic and comparitively logical system for the BERT data saving and generation. Unfortunately, it is not at a stage which is entirely finished yet, due to development difficulties. However, it is significantly closer as of now.
- *Connor*:
   + Per Ed's suggestion, gathered test cases for bert-qa-advanced.py for fine-tuning.
   
**4/25/20**:
- *Edward*:
   + Researched methods of implementing transformer qa based on QANet + Transformer XL information included in the above Stanford Article
   + Implementing pretrained wikipedia edited QA model that includes transformer-xl attention management to properly test against BERT
   + Rewrote pipeline for transformer-xl to accept test cases similar to bert-qa-advanced
   
- *Vincent*:
   + Upgraded the output of BERT's Q&A system to now output a CSV file containing 3 columns, one for the associated question, one for BERT's answer to that question, and one (Which is temporarily filled with "NULL"s) that will contain Transformer-XL's answer to the question, too.
   + Also implemented JSON file output, which is a more universal and consistant method of data storage. (The JSON contains an array of "QandA" objects, with a "Question", "BERT Answer", and "Transformer-XL" attribute for each of them.
   + Re-wrote old comments in the BERT Q&A for clarity-purposes.
- *Connor*:
   + Increased number of Q&A test cases from 18 to 50. 
   + Attempted to find a way to have the results of the Transformer-XL put into a specific column in the CSV file Vincent set up, but was unsuccessful.
   
**5/2/20**:
- *Edward*:
   + Additional implementation of xlnet on the transformer server. Additional research on using xlnet for QA. Basic implementation of sentence generation with xlnet/transformer-xl 
- *Vincent*:
   + Progress made towards xlnet Q&A's repair. Debugging unusual tuple structure issue with the output tokens.
   + Working on data visualization utilizing MatPlotLib 3.2
- *Connor*:
   + Worked on getting the skeleton of the research paper ready: abstract, works cited, and introduction.
