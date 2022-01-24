# drunktweets
Drunk tweets


## Dimensões
- DS1-Q1:
  - maxlen <- 38
  - max_words <- 7860
  - PCA <- 13
- DS1-Q2:
  - maxlen <- 38
  - max_words <- 4405
  - PCA <- 12
- DS1-Q3: 
  - maxlen <- 38
  - max_words <- 3080
  - PCA <- 9
- DS2:
  - maxlen <- 38
  - max_words <- 18391
  - PCA <- 50
- DS3:
  - maxlen <- 38
  - max_words <- 183919322
  - PCA <- ?? 


## Drink2Vec - DNN

### Resultados
[Planilha de resultados](https://docs.google.com/spreadsheets/d/1YFfZEjP8I3NONUvdPovOf1AwA6TEZbLdq_NaTMSlK90/edit#gid=450516288)

## Drink2Vec - SVM

### Resultados
[Planilha de resultados](https://docs.google.com/spreadsheets/d/1BU6NxRKoT2ozmOH1g9f6yRtO2qeXtiUs7J-4TCJ__IQ/edit#gid=628839925)


## Experimentos Ensemble

### Processo
- Para cada dataset, separei em treinamento e teste
- Roda individualmente cada classificador
- Armazenei a probabilidade da classe positiva para as instâncias de testes
- Cada instância de teste é composta de 3 probabilidades (uma associada a cada classificador)
- As instâncias de testes foram classificadas usando o algoritmo SVMPoly com cross-validation e os resultados apresentados

### Resultados
- https://docs.google.com/spreadsheets/d/1rxl28PVpPsGzPZiD6g1_CjXJbUSD4eOkXsKWFSfc_jM/edit?usp=sharing
- Aba WI + New DL + SVM (DL)


### Files
- `ensembles/ensemblev4/comparador/comparador_oficial_as_classifier_certo.R`
- `ensembles/ensemblev4/comparador/comparador_oficial_as_classifier_certo_ds2.R`

## Experimentos LSTM

### Files
Path: `exp4/svmpoly/lstm/ds1q1.R`


### Gerar embeddings
- DS1-Q1: `adhoc/exportembedding/ds1/lstm_q1.R`

- DS3:
  - `adhoc/exportembedding/ds3/lstm_10_epocas_v2.txt`

### Geradores média
- DS1-Q1: `adhoc/redemaluca/ds1/dados/q1_redemaluca_lstm_PCA.R`
- DS1-Q2: ``
- DS1-Q3: ``
- DS2: `exp4/svmpoly/lstm/ds2.R` **Verificar**
- DS3: `adhoc/redemaluca/ds3/dados/ds3_redemaluca_lstm_PCA.R`
  - `adhoc/redemaluca/ds3/ds3_representacao_with_lstm_pca_15.RData`



## Experimentos Bi-LSTM

### Info úteis
 - Folder: `ipmbilstm`
 - Comentar caret em `utils/getDados.R` 

### Passos
1 - Gerar embeddings
2 - Gerar média de cada tweet
3 - Classificar similar a `exp4/svmpoly/lstm/ds3.R`

### Gerar embeddings
- DS1-Q1: `ipmbilstm/exportembedding/ds1/bilstm_q1.R`
  - `ipmbilstm/exportembedding/ds1/q1/bilstm_10_epocas.txt`
- DS1-Q2: `ipmbilstm/exportembedding/ds1/bilstm_q2.R`
  - `ipmbilstm/exportembedding/ds1/q2/bilstm_10_epocas.txt`
- DS1-Q3: `ipmbilstm/exportembedding/ds1/bilstm_q3.R`
  - `ipmbilstm/exportembedding/ds1/q3/bilstm_10_epocas_new.txt`
- DS2: `ipmbilstm/exportembedding/ds2/bilstm_v2.R`
  - `ipmbilstm/exportembedding/ds2/bilstm_10_epocas.txt`
- DS3: `ipmbilstm/exportembedding/ds3/lstm_oficial.R`
  - `ipmbilstm/exportembedding/ds3/bilstm_10_epocas_v2.txt`

* AMI: 30062019RServer*

### Gerar média de cada tweet
- DS1-Q1: `ipmbilstm/ds1/dados/q1_redemaluca_bilstm_PCA.R`
  - `ipmbilstm/exportembedding/ds1/q1_representacao_bilstm_pca.RData`
- DS1-Q2: `ipmbilstm/ds1/dados/q2_redemaluca_bilstm_PCA.R`
  - `ipmbilstm/exportembedding/ds1/q2_representacao_bilstm_pca.RData`
- DS1-Q3: `ipmbilstm/ds1/dados/q3_redemaluca_bilstm_PCA.R`
  - `ipmbilstm/exportembedding/ds1/q3_representacao_bilstm_pca_new.RData`
- DS2: `ipmbilstm/ds2/dados/ds2_redemaluca_bilstm_pca.R`
  - `ipmbilstm/ds2/dados/ds2_representacao_bilstm_PCA_50.RData`
- DS3: `ipmbilstm/exportembedding/ds3/ds3_redemaluca_bilstm_PCA.R`
  - `ipmbilstm/exportembedding/ds3/ds3_representacao_with_bilstm_pca_15.RData`


*AMI: ubuntur3542206*

### Executar os classificadores

**SVM**
- DS1-Q1: `exp4/svmpoly/bilstm/ds1q1.R`
- DS1-Q2: `exp4/svmpoly/bilstm/ds1q2.R`
- DS1-Q3: `exp4/svmpoly/bilstm/ds1q3.R`
- DS2:  `exp4/svmpoly/bilstm/ds2.R`
- DS3: `exp4/svmpoly/bilstm/ds3.R`

* AMI: 30062019RServer*

**XGBOOST**
- DS1-Q1: `exp4/xgboost/bilstm/ds1q1.R`
- DS1-Q2: `exp4/xgboost/bilstm/ds1q2.R`
- DS1-Q3: `exp4/xgboost/bilstm/ds1q3.R`
- DS2:  ``
- DS3: ``


### BI-LSTM Menor
- `ipmbilstm/exportembedding/ds1/bilstm_q1_menor.R`
- `ipmbilstm/ds1/dados/q1_redemaluca_bilstm_PCA_menor.R`
- `exp4/svmpoly/bilstm/ds1q1_menor.R`


### Resultados
[Planilha de resultados](https://docs.google.com/spreadsheets/d/112byd2PSnWVh7KbdJP3AlDGEZne6a-5zVCjZrbNqXTg/edit?usp=sharing)

## Restore Database

Remover linhas
- SET @MYSQLDUMP_TEMP_LOG_BIN = @@SESSION.SQL_LOG_BIN;
- SET @@SESSION.SQL_LOG_BIN= 0;
- SET @@GLOBAL.GTID_PURGED='d2298455-xxxx-xxxx-xxxx-42010a980029:1-3413775';
- SET @@SESSION.SQL_LOG_BIN = @MYSQLDUMP_TEMP_LOG_BIN

[Fonte](https://help.poralix.com/articles/mysql-access-denied-you-need-the-super-privilege-for-this-operation)

## BERT

### Documentos
- Tutorial em code lab: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
- Tutoral em R: https://blogs.rstudio.com/ai/posts/2019-09-30-bert-r/
- Artigo: https://arxiv.org/pdf/1810.04805.pdf

### Menos importantes
- Doc oficial: https://github.com/google-research/bert


### Code Labs
- MarcosTutorial.ipynb: teste feito com DS1-Q2
- Marcos word_to_sentence_embedding.ipynb: IMDB

### Gerar CSV
- DS1-Q1: `bert/exportarcsv.R`
  - `bert/exp1.csv`
- DS1-Q2: `bert/exportarcsv.R`
  - `bert/exp2.csv`
- DS1-Q3: `bert/exportarcsv.R`
  - `bert/exp3.csv`
- DS2: `bert/exportarcsv.R`
  - `bert/ds2.csv`
- DS3: `bert/exportarcsv.R`
  - `bert/ds3.csv`

### Utilizar projeto
- https://github.com/MarcosGrzeca/berttestes


### AMI
- bertfunciona


### Resultados
[Planilha de resultados](https://docs.google.com/spreadsheets/d/1GYIbqFk9f_9-GQYBGADx9S_47XZIOl2ixR1eaqdAYFU/edit?usp=sharing)


## AMIS

### Usuários RStudio
- 30062019RServer: username:8787
- ubuntur3542206: username:8080
- bertfunciona: rstudio:8080


## CNN direta

### Files
- DS1-Q1: `drunktweets/cnndireto/ds1/cnn_q3.R`
- DS1-Q2: `drunktweets/cnndireto/ds1/cnn_q2.R`
- DS1-Q3: `drunktweets/cnndireto/ds1/cnn_q3.R`
- DS2: `drunktweets/cnndireto/ds2.R`
- DS3: `drunktweets/cnndireto/ds3.R`

### Resultados
[Planilha de resultados](https://docs.google.com/spreadsheets/d/1un4tlaz3wBAL0lCJ1xXHbrTQ8q64vqmlRqVFA__7_dA/edit?usp=sharing)

## Bi-LSTM direta

### Files
- DS1-Q1: `drunktweets/bilstmdireta/ds1/bilstm_q1.R`
- DS1-Q2: `drunktweets/bilstmdireta/ds1/bilstm_q2.R`
- DS1-Q3: `drunktweets/bilstmdireta/ds1/bilstm_q3.R`
- DS2: `drunktweets/bilstmdireta/ds2.R`
- DS3: `drunktweets/bilstmdireta/ds3.R`

### Resultados
[Planilha de resultados](https://docs.google.com/spreadsheets/d/1un4tlaz3wBAL0lCJ1xXHbrTQ8q64vqmlRqVFA__7_dA/edit?usp=sharing)