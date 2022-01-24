# Notes on Fraud Detection and Explainable AI
## Fraud Detection
### Notes
_Electronic Fraud_ is classified into two types of categories:
  - _Direct fraud_
    - For example: Money laundering, Salami technique, Employee embezzlement
    
    
  - _Indirect fraud_
    - For example: malware, phishing, identity theft
    
#### Direct Fraud
##### Money laundering
Converting illicit/illegal money into less suspicious money. This is done to conceal the source of where the money comes from. One technique is to transfer money to multiple accounts using complex transactions. 

##### Salami technique
Transfering a miniscule amount of money from a great amount of customer accounts, using f.ex. insiders in a bank.

#### Indirect Fraud 
##### Identity theft
Someone gains access to personal information to further gain access to services in for example banks.


### Resources
### Articles
- [Dzomira, Shewangu. (2015). Cyber-banking fraud risk mitigation - Conceptual model. Banks and Bank Systems. 10. 7-14.](https://www.researchgate.net/publication/282281102_Cyber-banking_fraud_risk_mitigation_-_Conceptual_model)
  - The aims for the article is to:
    - Critically review the forms of cyber banking fraud risks exposed in the financial sector. 
    - To propose a cyber-banking fraud risk ma- nagement model. 
 - [Application of Credit Card Fraud Detection: Based on Bagging Ensemble Classifier](https://www.sciencedirect.com/science/article/pii/S1877050915007103)

## Explainable AI

### Notes

#### Shapley values
[Wikipedia definition](https://en.wikipedia.org/wiki/Shapley_value)
Rettferdig fordeling av bidrag til prediksjon fra maskinlæringsmodell etter innsatts fra egenskapene (variabler, features)
-> Explains deviation from the average for given feature

#### [Contrafactual explanations](https://christophm.github.io/interpretable-ml-book/counterfactual.html)
- Can be hard to automatically generate based on one specific prediction with a big number of features.

#### LIME: Local interpratable model agnostic explenations
(not recommended by one lecturer -> why?)

### Unsorted
- Shapley values
- Lime 
- Mutual information (NN)
- Kontrafaktisk forklaring

### Books
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable, by Christoph Molnar ](https://christophm.github.io/interpretable-ml-book/)

### Articles, Reports, Books
- https://www.darpa.mil/program/explainable-artificial-intelligence
- [NTNU AI Lab project - EXAIGON: New project on explainable artificial intelligence](https://www.ntnu.edu/ailab/news)
- [COUNTERFACTUAL EXPLANATIONS WITHOUT OPENING THE BLACK BOX: AUTOMATED DECISIONS AND THE GDPR](https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf)
- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier
](https://arxiv.org/abs/1602.04938)
- [Open-source library provides explanation for machine learning through diverse counterfactuals
](https://www.microsoft.com/en-us/research/blog/open-source-library-provides-explanation-for-machine-learning-through-diverse-counterfactuals/)

### Tools
- [Captum - Model Interpretability for PyTorch](https://captum.ai/)
- [Interpret Community SDK](https://github.com/interpretml/interpret-community#models)

### Podcast episodes

- [TwiML: Data Innovation & AI at Capital One](https://twimlai.com/twiml-talk-147-data-innovation-ai-at-capital-one-with-adam-wenchel/)
- [TwiML: Explaining Black Box Predictions](https://twimlai.com/twiml-talk-73-exploring-black-box-predictions-sam-ritchie/)
- [TwiML: Fighting Fraud with Machine Learning at Shopify](https://twimlai.com/twiml-talk-60-fighting-fraud-machine-learning-shopify-solmaz-shahalizadeh)
- [Data Skeptic: Black boxes are not required](https://dataskeptic.com/blog/episodes/2020/black-boxes-are-not-required)

### Live events

- 14.mai.2020 Kl. 09:00–10:30 Tekna: Forklarbar Kunstig intelligens https://www.tekna.no/kurs/forklarbar-kunstig-intelligens--xai-39856/
  - [Link to stream](https://teams.microsoft.com/dl/launcher/launcher.html?url=%2f_%23%2fl%2fmeetup-join%2f19%3ameeting_ZGQ0NjU3OWYtMmY0Mi00Y2YwLWE0N2MtOWNmOWE4NDRmZjNm%40thread.v2%2f0%3fcontext%3d%257b%2522Tid%2522%253a%2522780b750e-d3a7-4fd6-9b5e-174dc7b56d9c%2522%252c%2522Oid%2522%253a%25228c17fadc-e4e9-46de-b04d-3106fc317f3e%2522%252c%2522IsBroadcastMeeting%2522%253atrue%257d%26anon%3dtrue&type=meetup-join&deeplinkId=b0e8aec6-a818-42d7-b875-fbcd08eb8c74&directDl=true&msLaunch=true&enableMobilePage=true&suppressPrompt=true). The stream might be unavailable later.

### Companies that work in this area
- https://www.sintef.no/en/explainable-ai/
- AI Lab is ramping up their research in this area.

### Other sources
- Kaggle competition https://www.kaggle.com/mlg-ulb/creditcardfraud

## ML Ops
### Tools 
- [MLFlow](https://mlflow.org/)
- [Pachyderm](https://pachyderm.io/)
- [Kubeflow](https://www.kubeflow.org/) (Does not support pytorch yet)

