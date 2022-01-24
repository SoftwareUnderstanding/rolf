#### 說明
以 SSD (Single Shot MultiBox Detector)作為車輛偵測與車種辨識的訓練模型。
參與車種圖片蒐集、資料集製作、模型訓練，以及圖片與影像的車輛偵測和車種分類預測。模型分類預測的準確率大約在74%。
車種辨識模型是基於<a href="https://github.com/zhreshold/mxnet-ssd.git">github之SSD 模型</a>，再根據本組製作的資料集與其他所需條件對模型進行相對應的修改予調整。 
詳細說明請見 Report - Detect Vehicle Types.pdf。

#### 執行環境
Python3 + Jupyter Notebook      

#### 目錄結構說明
* download.py: 從ImageNet蒐集13個類別之圖片集，存至ImageNet_Picture資料夾。
* ImageNet_ID.txt: 13個類別之ImageNet ID
* generate_VOCdevkit_format.py: 將資料集與本組標註的xml檔案轉換成VOCdevkit格式。
* ssd_training.ipynb: train the SSD model。
* model: the SSD model。(Need to extract ./model/data.zip and ./model/test_data.zip first.)
* test_result.zip: 測試圖片之預測結果(車種類別與score)，依據預測車種分類。
