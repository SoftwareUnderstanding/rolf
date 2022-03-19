### If you want an overall understanding of what I was doing, please refer to the PPT  
### "IML(presentation version).pptx"  file.  (You will only want to see this)  
My project for CIP(Thanks Cathay LAB department for giving me the opportunity! I'm very grateful.)  

## Folder layout:

1. Data Shapley with image : Trying to use Data Shapley to interpret a neural network   
Data : cats and dogs image  

2. InterpreML with relational data : Using Microsoft Interpret to interpret various algorithms  
Data : Bank Marketing dataset(Portugesse)  

3. Interpretable Machine Learning : Just a folder storing all the cores that my works use,including Interpret and DataShapley  
and some random tests mainly for debugging

## Papers I've read just for this project:  
# LIME  
https://arxiv.org/pdf/1602.04938.pdf  
My comment: A very innovative framework for interpreting a model via linear approximate

# SHAP

https://arxiv.org/abs/1705.07874  
My comment : Extraordinary theory to prove that Shapley Value(Game Theory) is the best method of attributing feature contributions. 

# Data Shapley
https://arxiv.org/pdf/1904.02868.pdf

Very recent(2019/5) paperwork  
My comment:  
Another innovative way to implement Shapley Value, SHAP uses Features as players(in game theory),while Data Shapley uses data instances as players.  
Data Shapley我認為真正進入到狹義的Global Interpretation，即可以得知＂預測模型＂(v.s.解釋模型)預測出來的結果大概是長什麼樣子  
Local interpretation我個人理解為＂得知預測模型怎麼預測(i.e.使用哪些features及相關的權重，且很好解釋)  

經過資料測試，發現只要跑稍為複雜的模型(bi-layer neural net work with 100 hidden units)，跑個1000筆資料就足以耗上好幾天的時間(on AWS,xlarge instance)   
Cons : 因為time complexity實在太高，只適合用在較珍貴的資料集上，來算出各資料點的價值  

# Microsoft Interpret
An overall interpret Opensource package by Microsoft(has SHAP/LIME/EBM) , a very nice visualization tool for interpretation  
Internal Engine : Explainable Boosting Machine(EBM) (GA2M Paper) : http://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf  
Github: https://github.com/microsoft/interpret  

My comment:  
屬於統計學習一派，Microsoft所開發出來（與想推銷的）內部引擎EBM為一個可解釋的＂預測模型＂，使用統計學習方法，以GAM為Baseline Model，將GAM回歸出來的殘差，再做一次交叉項(Pairwise Interaction effect)的回歸，期望把更多的殘差以交互作用來納入整體解釋能力。  
但個人感覺光是GAM可能就很難解釋了(Smoother function，你所train出來的個別函數是什麼等等)，加上GA2M又更難解釋，見仁見智。  


## 一些pdf檔案為我看過之後加上一些自己的筆記，有需要拿就自取吧

Last :  
a very good book for Interpretable Machine Learning(online,free), I learned very very much from this:
https://christophm.github.io/interpretable-ml-book/


# Version Control
### 8/19/2019  
## Upload InterpretML with relational data  
Using bank marketing datasets from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing  
Applies Microsoft InterpretML on it

## Upload Data Shapley with image (Most time consuming here)
Using datasets from https://github.com/Pranaw99/Image_Classification_CNN  
Applies Data Shapley on cat/dog images and compute the Data Shapley value  
這邊還沒完整測試過，跑一次Data Shapley需要大量時間，非常難debug  
所以正確性還有待觀察，持續補強中  
使用前且確保清除該路徑所有temp、.ipynb_checkpoints與__pycache__資料夾(執行DShap會產生)，不然發生一些奇妙的錯誤(這邊我花了好幾天百思不得其解之後才知道應該是快取問題)  

### 8/26/2019  
## Upload Data Shapley image with CNN (version3)  
using 250 data points  
Too time consuming, on AWS xlarge instance, it has to run at least 5 days or more  
I didnt finished training this model,but it should work. Feel free to use if you want

### 8/30/2019
Final upload, a small change to README, and add final powerpoint file
