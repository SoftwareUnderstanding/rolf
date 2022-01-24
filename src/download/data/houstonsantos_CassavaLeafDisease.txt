<p align="center">
    <img width="400" height="200" src="img/upepoli.png">
</p>

# Especialização em Ciência de Dados e Analytics

## Resumo

<p style='text-align: justify;'> Como o segundo maior fornecedor de carboidratos na África, a mandioca é uma cultura de segurança alimentar essencial cultivada por pequenos agricultores porque pode suportar condições adversas. Pelo menos 80% das fazendas familiares na África Subsaariana cultivam essa raiz amilácea, mas as doenças virais são as principais fontes de baixa produção. Os métodos existentes de detecção de doenças exigem que os agricultores solicitem a ajuda de especialistas agrícolas financiados pelo governo para inspecionar visualmente e diagnosticar as plantas. Isso sofre por ser muito trabalhoso, com baixo suprimento e caro. Diante disso, esta pesquisa tem como finalidade a construção de uma máquina preditiva, usando Redes Neurais Profundas (CNN) será possível realizar a identificação do patógeno acometido nas folhas da Mandioca. Para guiar nossa pesquisa foi utilizada a metodologia SEMMA. Os resultados obtidos com o algoritmo construído foram satisfatórios chegando ao resultado de 91% de assertividades nos diagnósticos dos patógenos. </p>

## Dados

* <p style='text-align: justify;'> O Banco de imagens utilizado é da National Crops Resources Research Institute (NaCRRI) em colaboração com o laboratório de IA da Makerere University, Kampala. Nele temos um total de 21397 imagens divididas da seguinte forma: </p>

    * Classe 0 - CBB
    * Classe 1 - CBSD
    * Classe 2 - CGM
    * Classe 3 - CMD
    * Classe 4 – Folha Saudável

<p align="center">
    <img width="500" height="400" src="img/picture1.png">
</p>

<p style='text-align: justify;'> Como podemos observar a classe 3 apresenta uma quantidade considerável em relação as demais, o desbalanceamento faz o modelo aprender mais sobre a classe predominante em relação as outras, esse tipo de problema é conhecido como classe rara. Para o problema da classe rara temos uma possível solução através da estratificação. Para realizar esse processo dados foram separados utilizando a técnica HoldOut de forma estratificada pela classe, mantendo suas proporções em cada separação; 70% (14977 para treino), 30% (6420 para teste), das imagens de treino 10% (642 para validação). Os dados de validação são usados ao final, com a intensão de simular um ambiente real de produção. </p>

## Modelo

Algumas informações de técnicas utilizadas na criação da Rede Neural Convolucional.

  * **Transfer Learning**
 
    * <p style='text-align: justify;'> O bjetivo da transferência de aprendizagem é melhorar a aprendizagem na tarefa alvo, aproveitando o conhecimento da tarefa de origem. Existem três medidas comuns pelas quais a transferência pode melhorar a aprendizagem. O primeiro é o desempenho inicial alcançável na tarefa alvo usando apenas o conhecimento transferido, antes que qualquer aprendizado adicional seja feito, em comparação com o desempenho inicial de um agente ignorante. O segundo é a quantidade de tempo que leva para aprender totalmente a tarefa alvo, dado o conhecimento transferido, em comparação com a quantidade de tempo para aprendê-la do zero. O terceiro é o nível de desempenho final alcançável na tarefa alvo em comparação com o nível final sem transferência. </p>

  * **Data Augumentation**

    * <p style='text-align: justify;'> A ideia de manipular imagens decorre de uma premissa básica das atividades que envolvem Deep Learning e a Aprendizagem de Máquina, que é a representatividade da amostra de dados utilizada para treinar um modelo. Quanto maior for a amostra e mais representativos forem os dados utilizados na etapa de treinamento, melhor será o desempenho do modelo ao classificar novos dados. Assim, ao rotacionar as imagens e redimensiona-las, aumentamos nosso espaço amostral gerando uma maior variabilidade de dados, fazendo com que nosso modelo aprenda características apresentadas sob outra "forma". </p>

<p align="center">
    <img width="800" height="600" src="img/picture3.png">
</p>
  
  * **Hiperparâmetros**
    
    * <p style='text-align: justify;'> ModelCheckpoint – Ao fim de cada epoch a função vai verificar o valor da val_loss, e comparar, salvando o modelo e seus pesos sempre que obtiver o menor valor para val_loss. </p>
    * <p style='text-align: justify;'> ReduceLROnPlateau – Checa o valor da val_loss ao final de cada epoch, se a mesma por duas epochs estagnar, irá reduzir a taxa de aprendizagem (learning rate). </p>
    * <p style='text-align: justify;'> EarlyStopping – Irá parar o treinamento se o valor da val_loss parar de diminuir por três epochs, entendendo que o modelo chegou em seu melhor ajuste, evitando assim o overfitting, na Figura 6 podemos verificar e entender melhor o funcionamento do EarlyStopping. </p>
    
<p align="center">
    <img width="300" height="200" src="img/picture2.png">
</p>

  * **Cross-Validation**
    
    * <p style='text-align: justify;'> A validação é um método estatístico que consiste em comprar algoritmos por meio de uma divisão de dados separados em dois segmentos, treino e teste essa separação dos dados dar-se o nome de folds (partes iguais), ao qual irá treinar e testar o modelo n (folds) vezes, com ajuda do scikit-learn usando StratifiedKFold podemos realizar esta separação de forma proporcional as classes </p>

<p align="center">
    <img width="400" height="300" src="img/picture4.png">
</p>


## Resultados

**HoldOut**
Consiste em uma separação simples do conjunto de dados entre treino e teste, onde o conjunto de dados pode ser separado em partes iguais ou não, uma proporção muito comum é 2/3 para treino e 1/3 para teste. Após o modelo ser criado e executado os testes são aplicados e a predição calculada. Para o experimento utilizou-se 70% para treino e 30% para teste. Nas figuras abixo podemos verificar a tabela com os resultados e a matriz de confusão. 

<p align="center">
    <img width="300" height="200" src="img/picture5.png">
</p>

<p align="center">
    <img width="350" height="300" src="img/picture8.png">
</p>

**Cross-Validation**
Na execução da validação cruzada foram utilizados 5 folds de forma estratificada, ou seja, mantendo a proporção das classes observadas. Neste método o teste não é executado ao final de cada epoch, como acontece quando utilizamos o HoldOut, e sim ao final do treinamento de cada fold, na tabela abaixo podemos observar os valores obtidos dos testes:

<p align="center">
    <img width="350" height="300" src="img/picture9.png">
</p>

<p align="center">
    <img width="350" height="300" src="img/picture11.png">
</p>

## Referências

<p>
[1] MORAIS, M. S.; MEDEIROS, E. V.; MOREIRA, K. A.; CAVALCANTI, M. S.; OLIVEIRA, N. T. Epidemiologia das doenças da parte aérea da Mandioca no Município de Alagoa Nova, Paraíba. SciELO Brasil, v.40, n.3, p.264-269, 2014.

[2] JUNIOR, F. A. G. Produtividade de Variedades de Mandioca em Diferentes Arranjos de Plantio, Épocas de Colheita, Fisiologia do Estresse e Déficit Hídrico. Bahia, UFRB 2018.

[3] A Cultura da Mandioca. Sistema de Produção Embrapa. Disponível em: <https://www.bibliotecaagptea.org.br/agricultura- novo/culturas-anuais/culturas-anuais-livros/> Acesso em: 15 abr. 2021.

[4] ARALDI, R.; SILVA, I. P. F.; TANAKA, A. A.; GIROTTO, M.; SILVA JUNIOR, J. F. Doenças Virais na Cultura da Mandioca. FAEF Revista Científica Eletrônica de Agronomia, n. 20 de 2011.

[5] Indução de Resistência em Manihot Esculenta Crantz ao Ataque de Vatiga Illudens. 
Disponível em: <https://cointer.institutoidv.org/inscricao/pdvagro/uploadsAnais/INDU%C3%87%C3%83O-DE-RESIST%C3%8ANCIA-EM-Manihot-esculenta-Crantz-AO-ATAQUE-DE-Vatiga-illudens.pdf> 
Acesso em: 15 abr. 2021

[6] OLIVEIRA, S. A. S.; SILVA, M. A.; RANGEL, M. A. S.; SANTOS, V. S.; RINGENBERG, R.; OLIVEIRA, E, J. Metodologia para Avaliação da Resistência da Mandioca à Bacteriose, Antracnose e Superalongamento.  
Disponível em: <https://ainfo.cnptia.embrapa.br/digital/bitstream/item/146099/1/boletim-78.pdf> Acesso em: 15 abr. 2021.

[7] TORREY, L.; SHAVLIK, J. (2009), Transfer Learning. Madison, WI: University of Wisconsin.

[8] MOURA, Karina. Data Science: Um estudo dos métodos no mercado e na academia, Porto Alegre: Universidade Federal do Rio Grande do Sul, 2018. Página 31.

[9] NOGUEIRA, D. Agile Data Mining: Uma metodologia ágil para o desenvolvimento de projetos de data mining. Porto, Portugal: Universidade do Porto, 2014, p.6-8.

[10] AZEVEDO, A. I. R. L.; SANTOS, M. F. KDD, SEMMA and CRISP-DM: a parallel overview. IADIS European Conference on Data Mining, 2008, IADIS. p.182–185.

[11] AMARAL, F. Introdução à Ciência de Dados: Mineração de Dados e Big Data: Rio de Janeiro: Alta Books; 2016.

[12] Agrupando conceitos e classificando imagens com Deep Learning. 
Disponível em:
<https://estevestoni.medium.com/agrupando-conceitos-e-classificando-imagens-com-deep-learning-5b2674f99539> Acesso em: 29 abr. 2021.

[13] TAN, M.; LE, V. Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Netwoks, 2020. Disponível em <https://arxiv.org/pdf/1905.11946.pdf>. Acesso 18 abril 2021.

[14] KOHAVI, R. A study of cross-validation and bootstrap for accuracy estimation and model selection. In: International joint Conference on artificial intelligence. [S.l.: s.n.], 1995. v. 14, p. 1137–1145.

[15] REFAEILZADEH, P.; TANG, L.; LIU, H. Cross-Validation. Azirona State University, 2008. Disponível em: <https://web.archive.org/web/20110905044421/http://www.public.asu.edu/~ltang9/papers/ency-cross-validation.pdf>. Acesso em: 19 abril 2021.

</p>
