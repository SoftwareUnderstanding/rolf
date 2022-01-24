# Machine Learning Guild - NLP Practicum

Textbook: 
- Speech and Language Processing (SLP): https://web.stanford.edu/~jurafsky/slp3/
- SLP YouTube Videos: https://www.youtube.com/playlist?list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm

# Setup and Installation
- Note: Repository has been built for Windows computers. Some packages may be incompatible with Macs.
- Environment setup instructions: https://www.youtube.com/watch?v=sUUWLBmj7Xc&feature=youtu.be
- Config.ini: You will need to change two lines in the config.ini file under [USER]. Change the text following "USERNAME:" to your username and "RAW_DATA:" to the file path to the raw data folder within the repository.

# LESSONS

### 0. Configuration (Pre-work)
*  Topics: course overview, git bash, python config.ini files, conda virtual environments
*  Technology: git bash, configparser, conda
*  Homework: use the command line to search data among 1000's of server configuration files

### 1. Text Extraction
*  Topics: Extract text from docx, pdf, and image files
*  Technology: docx, PyPDF2, pdfminer.six, subprocess, pytesseract
*  Homework: structure the annual reports into sections

### 2. Text Preprocessing
*  Topics: lemmatization, POS tagging, dependency parsing, rule-based matching
*  Technology: SpaCy
*  Prework: Read section 2.1-2.4 SLP and/or 2.1-2.5 SLP videos
*  Supplementary Material: watch lesson_2a_automation videos

### 3. Phrase (collocation) Detection
*  Topics: acronyms, POS phrases, phrase dectection
*  Technology: SpaCy, gensim
*  Prework: Read section 8.1-8.3 SLP and chapter 5 Collocations (https://nlp.stanford.edu/fsnlp/promo/colloc.pdf)
*  Supplementary Material: watch lesson_3a_databases videos

### 4. Text Vectorization (count-based methods)
*  Topics: vector space model, TFIDF, BM25, Co-occurance matrix
*  Technology: scikit-learn
*  Prework: Read section 6.1-6.6 SLP
*  Supplementary Material: watch lesson 4a_object_oriented_python

### 5. Dimensionality Reduction
*  Topics: PCA, latent semantic indexing (LSI), latent dirichlet allocation(LDA), topic coherence metrics
*  Technology: scikit-learn, gensim
*  Prework: Read TamingTextwiththeSVD (ftp://ftp.sas.com/techsup/download/EMiner/TamingTextwiththeSVD.pdf)

### 6. Word Embeddings 
* Topics: Word2Vec, GloVe, FastText, ELMO, ULMFit 
* Technology: scikit-learn, gensim
* Prework: Read section 6.8-6.13 SLP, Efficient Estimation of Word Representations in Vector Space (https://arxiv.org/pdf/1301.3781.pdf), & Distributed Representations of Words and Phrases
and their Compositionality (https://arxiv.org/pdf/1310.4546.pdf)

### 7. Text Similarity
*  Topics: cosine similarity, distance metrics, l1 and l2 norm, recommendation engines
*  Technology: scikit-learn, SpaCy, gensim
*  Prework: Read section 2.5 SLP and/or 2.1-2.5 SLP videos

### 8. Document Classification
*  Topics: document classification, evaluation metrics, machine learning experiment set-up
*  Technology: scikit-learn
*  Prework: Read chapter 4 & 5 SLP
*  Supplementary Material: watch lesson 8a_pipeline_and_custom_transformers


# SUPPLEMENTARY MATERIAL

### 2a. Automation (Optional)
*  Topics: automate the process to collect data from https://www.annualreports.com
*  Technology: requests, Jupyter Notebooks, BeautifulSoup, Scrapy
*  Homework: automate the process to identify and download company 10-K annual reports

### 3a. Databases
*  Topics: use sqlalchemy to create and populate a database, locally and on AWS
*  Technology: sqlalchemy, sqllite, AWS RDS (MySQL)
*  Homework: create and populate a database with sqlalchemy

### 4a. Object Oriented Python
*  Topics: reconstruct scikit-learn's CountVectorizer codebase
*  Technology: scikit-learn, object oriented Python

### 8a. Pipelines and Custom Transformers
*  Topics: capture, format, and send logging messages to a variety of output. Exception Handling. Create an executable of a python package for deployment
*  Technology: scikit-learn, logging, python exceptions, pyinstaller, argparse