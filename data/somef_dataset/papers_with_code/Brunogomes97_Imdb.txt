# Imdb-Wiki : Idade e gênero 
Extração e Classificação da base do imdb para idade e gênero




Arquivos necessários para a extração e classificação IMDB-Wiki Dataset.

- Rápida definição dos arquivos:

-> [IMDB_analise](https://github.com/Brunogomes97/Imdb/blob/master/IMDB_Analise.ipynb): Analisando o dataset antes e após a realização de processamento, métodos e afins.

-> [ImagePredict](https://github.com/Brunogomes97/Imdb/blob/master/ImagePredict%20.ipynb): Testando a rede pré-trinada no dataset Wiki.

-> [WebcamPred](https://github.com/Brunogomes97/Imdb/blob/master/WebcamPred.py): Teste da rede na Webcam(Predict).

-> [Imdb_mat](https://github.com/Brunogomes97/Imdb/blob/master/imdb_mat.py): Script para gerar base de dados processada.

-> [functions](https://github.com/Brunogomes97/Imdb/blob/master/functions.py): Funções auxiliares de préprocessamento.

-> [imdb_train](https://github.com/Brunogomes97/Imdb/blob/master/imdb_train.py): Script de treino da rede.

-> [TensorBoard](https://github.com/Brunogomes97/Imdb/tree/master/Tensorboard): Pasta contendo todos os dados equivalentes aos treinos em modelos de redes neurais.(No momento somente WRN)

->[utils](https://github.com/Brunogomes97/Imdb/blob/master/utils.py): Métodos auxiliares


Para entender e criar um método de carregar e filtrar erros na base veja o Notebook IMDB_analise. Para o treinamento, há outras importações de outros repositórios a serem feitas.




 - Treinamento
 
 Para este treinamento foi utilizada inicialmente uma WideResNet com resolução de imagem (64 x 64), aliado a usos de bibliotecas de processamento de imagens rodando por baixo do treinamento. 
 
- Modelos de Rede
 
  ->WideResNet:
  
  -> Def: Arquitetura de rede neural baseada em blocos de convolução e normalização de Batchs. Investe na largura das rede para diminuir a profundidade de camadas. O modelo adotado neste teste foi de 16 camadas de convolução.
  
  -> Artigo: https://arxiv.org/abs/1605.07146
  
  -> Implementação: O porte para o framework Keras pode ser encontrada neste [repositório](https://github.com/asmith26/wide_resnets_keras)
  
  -> Implementação: A modificação da rede para estimar idade e gênero ao mesmo tempo pode ser encontrada [aqui](https://github.com/yu4u/age-gender-estimation/blob/master/wide_resnet.py)
  
  
  
  
  
- Métodos de processamento de Imagem(Augmentation):

  -> Mixup: Técnica de processamento de imagem que "mixa" imagens para aumentar a generalização dos testes. O mixup se baseia em interpolação linear de imagens para criar mais dados para treino.
  
   Informações sobre: https://www.dlology.com/blog/how-to-do-mixup-training-from-image-files-in-keras/ 
   
   Artigo: https://arxiv.org/pdf/1710.09412.pdf
   
   Port para o Keras pode ser encontrado [aqui](https://github.com/yu4u/age-gender-estimation/blob/master/mixup_generator.py)
  
  -> Random Eraser: Técnica de processamento que visa gerar várias imagens com retângulos randômicos por cima de uma mesma imagem. A ideia é apartir da oclusão de informações conseguir evitar o Overfitting da rede.
  
  Artigo: https://arxiv.org/abs/1708.04896
          
  Repositório: https://github.com/zhunzhong07/Random-Erasing
          
  Port para o Keras pode ser encontrado [aqui](https://github.com/yu4u/age-gender-estimation/blob/master/random_eraser.py)
  
  
  
  
- Callbacks utilizados:

   ->Checkpoint: Salva os pesos com os melhores desempenhos ao longo das epocas de treinamento
   
   ->LRSchedule: Diminui a Learning Rate da rede conforme a quantidade de épocas vai passando.
   
   ->Tensorboard: Ferramenta de visualização e depuração da rede.
   
*O modelo pré-treinado produzido pela WRN (Treinado no IMDB e testado na WIKI) pode ser baixado [aqui](https://1drv.ms/u/s!AtjWonwDFRIckHgtAl2R9ykrhNpt?e=7EWO8x)


Modelos Futuros a serem testados:

  WRN resolução 128x
  
  VGG16
  
  VGG16 Finetuning(com outras bases)
  
Outras bases:

  FaceScrub
  UTK dataset

          
  





