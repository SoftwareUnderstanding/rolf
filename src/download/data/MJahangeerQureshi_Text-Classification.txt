# Text-Classification
This repo has 2 different ML algorithms tackling text classification.
* Semantic Subword Hashing(https://arxiv.org/pdf/1810.07150)
* ULMfit(https://arxiv.org/pdf/1801.06146)

Both have State of the Art preformances on several datasets. As a rule of thumb it is best to use Semantic Subword Hashing on smaller corpora while using ULMfit on larger ones.

# To run the Scripts firstly setup the enviroment by running the following commands
```
pip install -r requirments.txt
pip install -U spacy
python -m spacy download en
python -m spacy download en_core_web_lg
python nltk_packages.py
```
### To test out Semantic Subword Hashing run
```
python train_ssh.py
```
### To test out ULMfit run
```
python train_ulmfit.py
```
