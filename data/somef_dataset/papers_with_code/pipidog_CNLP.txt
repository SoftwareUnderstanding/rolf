<p align="center">
  <img src="./img/CNLP_logo.png" width="600px"/>
</p>

# 歡迎來到Project CNLP

CNLP是一個基於Python以及深度學習(Keras & Tensorflow)來完成文本分類以及情緒預測的中文自然語言處理包（同時支援繁體與簡體）。 它能夠幫助使用者使用快速的完成高頻詞統計，建構深度學習模型，以及預測未知數據。這個工具包總共含有三個主要的模組：  

* 高頻詞統計(NLP_Stat)：這個模組可以幫助使用者快速的載入數據，清洗文本資料，中文分詞，高頻詞統計，並將高頻詞以圖表方式呈現，幫助使用者快速的對文本數據的特性有個大致的了解。

* 建構深度學習模型(NLP_Model)：這個模組能夠幫助使用者快速進行大量的文本數據預處理，將中文文本轉換成相應的數值向量。此模組還能夠幫助使用者快速的建構一個深度RNN模型，包括單向以及雙向的RNN，並支援多種RNN cell，包括Simple RNN, GRU,以及 LSTM。此外CNLP亦支援近年來流行的n-gram CNN，在模型建構完成後亦可輕易地調用多種優化器(SGD, PRMprop, Adam)進行模型訓練以及測試。

* 未知數據預測(NLP_Pred): 這個模組能夠幫助使用者快速的調用已經訓練好的深度學習模型，並使用該模型對未知的數據進行數據清洗以及結果預測。

# 背景介紹
文本分類以及情緒分析是自然語言處理以及深度學習中常見的問題。目前已經有多種針對英文文本數據的自然語言處理工具包，但是對於中文的工具包仍然十分缺乏。因此建構了這個簡單的工具包來快速的搭建深度學習模型做文本數據分析。本工具包的目的並不在提供強大的擴展性，而是希望在不失基本的彈性的情況下，讓使用者盡可能使用少量的命令來處理大多數的文本分類問題。    

中文數據和英文數據很大的不同在於切詞(tokenize)，因為英文裡面，每個單字會以一個空格做區隔，但是在中文裡面，中文詞彙之間並沒有類似的區隔符，這導致了要把中文詞彙做出正確的分割十分困難。為了解決這個問題，在CNLP中我們使用了jieba作為中文切詞引擎 (https://github.com/fxsjy/jieba) ,jieba是一個相當優秀的python中文切詞工具，除了能自動切詞之外，使用者也可依據自己的需求來增加或刪除詞彙，並調整詞頻高低。在CNLP中，你能夠使用簡單的函數就調用jieba處理大多數的中文文本場景。

在深度學習方面，CNLP能夠對文本數據進行預處理，將文本數據轉換成訓練所需的向量形式，並使用了Keras以及Tensorflow來進行深度學習。裡面已經預先架構好了一個能適用大多數場景的深度學習RNN模型，使用者能夠依據自己的需求來調整模型相關的參數。在訓練完畢之後，CNLP亦可調用已經訓練好的深度學習模型來對未知的數據進行預測。

由於中文自然語言處理的數據極其缺乏，在CNLP裡面也整理了兩個數據集供使用者測試。第一個數據集是來自攜程網的酒店評論數據，這個數據量偏少，正負評各約一千則（如果有更大的中文情緒分析數據集也請來信告知，讓我加入其中），作為深度學習之用太小了，但主要是供大家測試之用。第二個數據集是來自北京清華大學的THUCNews，這個數據集包含十四個類別，超過七十萬則新聞，整個數據超過1.5G，太大了，不適合做為測試之用，因此我們採用了網路上提供的另一個簡化版本 (https://github.com/gaussic/text-classification-cnn-rnn) ,在這個簡化版本中，共有十個類別，50000條訓練數據，5000條驗證數據，以及10000條測試數據。以上兩個數據集已經完成了基本的數據清洗，並且上好了相應的標籤。使用者可以在程式裡面直接使用不用再做任何處理。此外為了滿足不同中文使用者的需求，兩個數據集都提供了簡體以及繁體中文的版本。

考量到使用者可能會有自己的需求，因此CNLP的設計是，每一步執行完後都會將結果儲存成pickle檔，以供日後調用(亦可以參數方式輸出)。因此除了使用CNLP來進行完整的分析外，亦可使用CNLP來進行部分的預處理，搭配自己的程式使用，增加了CNLP的彈性。

# 深度學習模型
為了方便使用者快速建立深度學習模型，在CNLP中提供了兩種典型的文本分類模型，分別是RNN以及n-gram CNN。

## RNN 
使用RNN處理NLP是一種相當自然的做法，每個詞向量先被表示成孤熱編碼形式（one-hot)，而後被送入word embedding層進行壓縮 。在這裡我們並沒有使用任何Pretrained word embedding，所有的embedding參數都是透過數據本身訓練得出的。而後這些被壓縮過後的詞坎入向量會送往一個深度RNN層（深度由使用者定義），而RNN層輸出的向量會再送往一個dense layer最後送到機率輸出層，模型的示意圖表示如下：

<p align="center">
<img src="./img/model_rnn.png">
<em> An illustion of RNN with depth=3</em>
</p>

在本模型中，使用者需要定義的是embedding層的大小( i.e. 輸入向量要被壓縮的維度)，RNN層要使用何種cell (Simple RNN, LSTM, GRU)，每個cell中neuron的數目，RNN層的深度 (本圖中，深度為3)，以及最後用來執行fine tuning的de nse layer的大小。圖中最後一層dense layer 是機率輸出層，所以大小和分類的類別數相同，因此使用者無需自訂。另外每個RNN cell中都採用了dropout，所以使用者亦需給定dropout rate。

一般來說，深度RNN計算量較大，且較不易訓練， 也較容易產生性能飽和，但是如果文本的特性具有相當的長程關聯性 (i.e. 文本前後間的關聯性高，難以靠識別幾個關鍵字就抓出主題)，則使用 RNN是必要的。不過在一般的情況下，我們建議使用者優先考慮n-gram CNN。

## n-gram CNN
n-gram CNN是最年來開始興起的NLP文本分類方法，最早是2014年由NYU的[Yoon Kim (arXiv:1408.5882)](https://arxiv.org/abs/1408.5882) 所提出，該論文提出後立刻引起了廣泛的注意，在短短三年間就累積了超過兩千次的引用率，其後有諸多改良版本被提出，在這CNLP中我們採用的是由UT Austin的 [Y. Zhang & B. Wallace (arXiv:1510.03820)](https://arxiv.org/abs/1510.03820) 所提出的架構，該架構可表示如下：

<p align="center">
<img src="./img/model_cnn.png">
<em>illustion of n-gram CNN with three different n</em>
</p>

在本模型中，使用者需要給定的參數包括要使用哪些 n-gram，n-gram給定的越多，上圖中的卷積層分支就會越多，一般建議可從n_gram = [2,3,4])開始嘗試， 則會如上圖共有三個分支，分別執行n = 2,3,4的卷積。最後送到global max pooling層（使用global max pooling而不用一般的max pooling是近年來較流行的做法, e.g. [arXiv:1312.4400](https://arxiv.org/abs/1312.4400))，諸多實驗表明這可以減少overfitting，提升計算性能，同時又不影響模型表現，這也是Z&W的論文中所建議採取的方法。因此CNLP中我們也採用此法，減少輸入參數與使用者的困擾，而後經由一個dropout送到輸出層。同樣的，該輸出dense層的大小必須等同於分類的類別數，因此無需使用者自訂大小。

從上面的討論可以發現，n-gram CNN所需要給定的參數相當少，另外計算速度也相當快，在許多測試中的表現亦優於RNN，因此一般的情況下，我們建議使用者優先嘗試n-gram CNN。

# 系統要求
使用CNLP前，必須先裝下列Python套件:
 * numpy, matplotlib, seaborn, pandas, sklearn, nltk (若你使用的是anaconda的發行版本，這些套件已經預設安裝了). 

 * jieba, tensorflow, keras, tqdm 

 * graphviz, pydot ([非必要，僅用於模型以png格式輸出功能](https://keras.io/#installation))

 # 安裝
 將本repo下載後，解壓縮，將/CNLP加入你的Python path中

 # 範例與數據集
 CNews10以及Hotel評論數據及可以在此處下載(~150 MB)  
 [Download CNNew10 & Hotel Review](https://my.pcloud.com/publink/show?code=XZ4loB7Z4XtW9zxRlS7LgWWVEuQmc8KrA5DX)

 # 快速使用說明
 * 要執行一個計算，只需要實例化一個CNLP物件，並且依照下面所附的步驟執行計算即可。

 * 執行的步驟順序不可交換，但是CNLP會將每個方法所產生的關鍵數據以暫存檔的形式存入output資料夾中，當下次執行後續的步驟時，並不需要將前面的步驟重新執行一次，CNLP會自動讀取前一個步驟的暫存檔，簡化了測試的流程。
 
 * CNLP在每個方法被執行後可以選擇回傳關鍵數據讓使用者可以搭配自己的程式使用，若你沒有這樣的需求，則並不需要將變數回傳。

 * 關於每個方法中相關參數的意義，請見doc資料夾中的文件。

 * procedure of a task:
	 * 高頻詞統計:
	 <p align="center">
	  <img src="./img/stat_usage.png">
	 </p>

	 * 建構深度學習模型:
	 <p align="center">
	  <img src="./img/model_usage.png">
	 </p>
 
	 * 預測未知數據:
	 <p align="center">
	  <img src="./img/pred_usage.png">
	 </p>

## 如何使用自訂的深度學習模型
如果預設的深度學習模型不符合需求，使用者可以跳過model.build_cnn或是model.build_rnn，改使用Keras定義自己的深度學習模型。定義好之後,只要使用Keras將模型存檔為work_dir/output/model.h5即可，之後的model.train以及model.test方法會自行讀取該檔。