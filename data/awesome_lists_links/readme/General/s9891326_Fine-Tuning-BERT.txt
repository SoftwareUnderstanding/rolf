# OpView情緒訓練工具
本專案為針對OpView資料內容，進行情緒分析(Fine-Tuning BERT)，並提供Makefile輔助訓練。

## outline
- [目的](#目的)
- [需求](#需求)
- [Requirements](#requirements)
- [情緒對照表](#情緒對照表)
- [輸入資料範例](#輸入資料範例)
- [輸出資料格式](#輸出資料格式)
- [Quick start](#quick-start)
- [模型說明](#模型說明)
- [功能說明](#功能說明)
    - [情緒模型](#情緒模型)
        - [訓練模型](#訓練模型)
        - [驗證模型](#驗證模型)
        - [展示模型](#展示模型)
    - [混合訓練集](#混合訓練集)
- [訓練工具使用步驟](#訓練工具使用步驟)
- [部屬說明](#部屬說明)
- [實驗紀錄](#實驗紀錄)
    - [比較learning_rate對各模型的影響](#比較learning_rate對各模型的影響)
    - [比較各模型的架構](#比較各模型的架構)
    - [比較字詞長度與batch長度的關係](比較字詞長度與batch長度的關係)
    - [Siege_test](#siege_test)
    - [模型容量比較](#模型容量比較)
- [參考資料](#參考資料)
- [FAQ](#faq)
- [Future_work](#future_work)

## 目的
- 減少訓練模型的難度，以及更新原有的框架(TensorFlow 1 -> 2)
- 使用較少的資料訓練模型，藉此來降低訓練模型的難度以及訓練時間
- 設計持續更新模型的機制

## 需求
1. 訓練工具需求
    - 輸入資料集(訓練集、測試集)，輸出訓練好的BERT模型(savedModel格式)
2. 模型需求
    - 輸入文章字串，輸出機率分數和預測分類

## Requirements
- ubuntu == 18.04
- python >= 3.6.9
- [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)
    <details>
    <summary>安裝步驟</summary>
    
    ```
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
    sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
    ```
    </details>
- Nvidia 顯卡驅動(nvidia-smi)
    <details>
    <summary>安裝步驟</summary>

    ```
    安裝步驟
    1. sudo apt install ubuntu-drivers-common
    2. ubuntu-drivers devices - 查看顯卡型號和推薦安裝驅動版本號，並選擇想安裝的版本
    2. sudo apt install nvidia-driver-455 - 安裝驅動
    3. 重新啟動 (sudo reboot)
    4. nvidia-smi - 檢查GPU狀態
    ```
    </details>

## 情緒對照表
```json
{
  "0": "正面",
  "1": "負面",
  "2": "中立"
}
```

## 輸入資料範例
- 檔案目錄 : datasets/old_sentiment/
- 訓練資料`train.tsv`與驗證資料`test.tsv`皆為`tsv格式`，並且以`\t`作為分隔。

text|label|
----|-----|
噓:皇民是日本人的意思喔，罵窩們皇民是在稱讚吧|1
推:顏色好美！|0
推:一銀那麼好借怎麼都不借我|2

## 輸出資料格式
- 即各個文章字串的機率分數，抓最大值來代表情緒
- e.g : [0.00403965 0.98463815 0.01132222]，最大值為**0.98463815**，index=**1**，對應到對照表，即代表**負面**情緒
```
[[0.00403965 0.98463815 0.01132222],
 [0.9897307  0.00142236 0.00884701],
 [0.00303223 0.9747216  0.02224614],
 ...
 [0.09621345 0.46099612 0.4427904 ]
 [0.0937982  0.21287207 0.6933297 ]
 [0.23805933 0.00547114 0.75646955]]
```

## Quick start
- 確認資料集(dataset/old_sentiment/)是否存在，且包含train.tsv、test.tsv

#### 建立images
- 利用docker file建立images，並掛載當前目錄到images內
```bash
make build_images
```

#### 訓練模型
- 透過建立好的docker image進行run的操作。可以指派要使用的GPU數量
```bash
make run_fine_tuning_sentiment
```

#### 驗證模型
- 透過建立好的docker image進行run的操作。可以指派要使用的GPU數量
```bash
make run_test_sentiment
```

#### inference模型
- 透過建立好的docker image進行run的操作。可以指派要使用的GPU數量
```bash
make run_inference_sentiment
```

#### 快速部屬
- 訓練好想要的模型後進行部屬，讓模型能快速讓人使用。
```bash
make deploy_model
```

## 模型說明
- 參數model_type有三種類型
    - [custom、origin](https://huggingface.co/transformers/model_doc/bert.html#tfbertforsequenceclassification)
        - 從transformers上抓pre-trained model
        - custom會在最後一層增加一層Dense進行softmax並進行訓練
    - [tf-hub](https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/3)
        - 從tensorFlow-hub上抓pre-trained model
        - 會在最後一層增加一層Dense進行softmax並進行訓練
- 儲存格式比較
    - `savedModel`
        - 需要1.4G左右的GPU Memory
        - 用於test、demo、載入舊模型訓練新模型
    - `h5 -> savedModel`
        - 一開始存成h5再重新load_weights再存成savedModel
        - 只需要用400M左右的GPU Memory
        - 用於deploy
    - `tflite`
        - [Ref.](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/model_servers/main.cc)
        - 提供給穿戴式裝置的模型，但TensorFlow官方說為實驗性質，導致無法順利部屬到tf-serving上
    - `h5 -> savedModel -> tensorRT`
        - [Ref.](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)
        - default parameter: `precision_mode`: `fp16`、`minimum_segment_size(n)`: `3`、`maximum_cached_engines(c)`: `1`
        - 最少需要約880M左右的GPU Memory用於啟動，tensorRT還會自己吃掉一些Memory(可以設定)
        - 第一次需要初始化模型，需要約1分鐘左右的時間

## 功能說明
### 情緒模型
- 使用Command Line Interface(CLI)串接輸入參數 
```
python core/run_sentiment.py --help
```

<details>
<summary>args</summary>

```
Usage: run_sentiment.py [OPTIONS]

Options:
  --data_dir TEXT            The input data dir. Should contain the .tsv files
                             (or other data files) for the task.  [required]

  --task_name TEXT           The name of the task to train.  [required]
  --output_dir TEXT          The output directory(savedModel) where the model
                             checkpoints will be written.  [required]

  --deploy_dir TEXT          If direct save the savedModel,will too large.
                             Because there default save the optimizer.So we
                             save to .h5 format, and convert to savedModel,
                             that will not save optimizer in savedModel.And
                             used this variable to deploy to tensorFlow
                             serving

  --learning_rate FLOAT      The initial learning rate for Adam.  [default:
                             2e-05]

  --num_epochs INTEGER       Total number of training epochs to perform.
                             [default: 3]

  --dropout_rate FLOAT       Discard some neuron in the network to prevent
                             cooperation between feature. E.g., 0.1 = 10% of
                             neuron.  [default: 0.1]

  --batch_size INTEGER       Total batch size for training.  [default: 32]
  --max_seq_length INTEGER   The maximum total input sequence length after
                             WordPiece tokenization. Sequences longer than
                             this will be truncated, and sequences shorter
                             than this will be padded.  [default: 128]

  --do_train BOOLEAN         Whether to run training.  [default: False]
  --do_test BOOLEAN          Whether to run test on the dev set.  [default:
                             False]

  --do_inference BOOLEAN     Whether to run the model in inference mode on the
                             test set.  [default: False]

  --save_format LIST         Default is ["savedmodel", "tensorRT"]. Input have
                             "savedmodel" will save savedmodel with h5 convert
                             to savedmodel, when input have "tensorRT" and
                             "savedmodel", will save converted savedmodel to
                             tensorRT, if just "tensorRT" will save origin
                             savedmodel to tensorRT, input have "tflite" will
                             save tflite format, but that is not correct work.
                             [default: savedmodel, tensorRT]

  --model_type BOOLEAN       Which model type do you want to use.
                             Default is "custom", can use "custom"、"tf-hub"、"origin
                             [default: custom]

  --model_name TEXT          Please input "bert-base-chinese"、"ckiplab/bert-
                             base-chinese"、"ckiplab/albert-base-chinese"、
                             "ckiplab/albert-tiny-chinese",to set pretrained
                             model name.  [default: bert-base-chinese]

  --load_model_dir TEXT      If you want load old model(savedModel) to train
                             the new model, please set where to load the old
                             model dir.

  --use_dev_dataset TEXT     Whether to evaluation the other dataset when
                             do_test times. The dataset name is
                             latest_test.tsv  [default: False]

  --mix_number INTEGER       How many training dataset to mix? If not set, the
                             old datasetwill be mixed according to the number
                             of tags in the new dataset  [default: 0]

  --help                     Show this message and exit.
```
</details>

#### 訓練模型
- 讀取**data_dir**參數下的檔案(train.tsv、test.tsv)，進行Fine-Tuning BERT多類別任務，最後儲存[模型報告](#模型報告)
- shell args 必帶參數
```
--task_name=$(TASK_NAME) \
--do_train=True \
--save_format=$(SAVE_FORMAT) \
--model_type=$(MODEL_TYPE) \
--batch_size=16 \
--data_dir=$(DATA_DIR) \
--model_name=$(MODEL_NAME) \
--output_dir=$(OUTPUT_DIR) \
--deploy_dir=$(DEPLOY_DIR)
```
- 目前測試過的模型checkpoint(ckiplab: 中研院)，
    - `bert-base-chinese`
    - `ckiplab/bert-base-chinese`
    - `ckiplab/albert-base-chinese`
    - `ckiplab/albert-tiny-chinese`
- 客製化模型Input、Output藉此來符合舊規格
<details>
<summary>Code example</summary>

```python
import tensorflow as tf
from transformers import TFAutoModel
from absl import flags

FLAGS = flags.FLAGS

def create_model(self):
    """Build BERT model. Custom Input Layer(input_ids、input_mask、segment_ids) and Output Layer(Dropout、Dense)"""
    input_word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
                                           name="input_ids")
    input_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
                                        name="segment_ids")
    # because ckiplab/models is use Pytorch develop so need to add 'from_pt' args
    model = TFAutoModel.from_pretrained(
        pretrained_model_name_or_path=self.model_name,
        from_pt=True if "ckiplab/" in self.model_name else False,
        num_labels=len(self.label_list)
    )
    model._saved_model_inputs_spec = None
    sequence_output = model([input_word_ids, input_mask, segment_ids])
    out = tf.keras.layers.Dropout(FLAGS.warmup_proportion)(sequence_output.pooler_output)
    out = tf.keras.layers.Dense(
        units=len(self.label_list),
        activation="softmax",
        name="probabilities"
    )(out)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_mask, segment_ids],
        outputs=out,
        name=self.task_name)

    model.summary()

    return model
```
</details>

- 如果是要用load_model的訓練方式，則需準備好新的訓練集，預設命名為`new_train.tsv`，存放在`dataset/old_sentiment`下，並設定讀取舊模型的路徑
```shell script
--task_name=$(TASK_NAME) \
--do_train=True \
--save_format=$(SAVE_FORMAT) \
--data_dir=$(DATA_DIR) \
--load_model_dir=$(LOAD_MODEL_DIR) \
--mix_number=$(MIX_NUMBER) \
--model_name=$(MODEL_NAME) \
--output_dir=$(OUTPUT_DIR) \
--deploy_dir=$(DEPLOY_DIR)
```

##### 模型報告
- 訓練完模型後，儲存的模型歷史紀錄，以及驗證資料的confusion matrix，用來觀看模型好壞
<details>
<summary>summary 範例</summary>

```
        - epoch:         [0, 1]
        - params:        {'verbose': 1, 'epochs': 2, 'steps': 417}
        - loss:          [0.40817883610725403, 0.21885691583156586]
        - accuracy:      [0.8476840853691101, 0.9230538010597229]
        - val_loss:      [0.2668299973011017, 0.2991047501564026]
        - val_accuracy:  [0.9088318943977356, 0.9017093777656555]
        - report:      
              precision    recall  f1-score   support

           0     0.9366    0.9131    0.9247      1036
           1     0.8828    0.9300    0.9058      1328
           2     0.8687    0.8345    0.8513      1142

    accuracy                         0.8939      3506
   macro avg     0.8960    0.8925    0.8939      3506
weighted avg     0.8941    0.8939    0.8936      3506

        - total_time:    621.076s
```
</details> 

#### 驗證模型
- 透過**output_dir**參數來指定模型位置，並進行**keras.load_model**的動作，再讀取**data_dir**裡的test.tsv進行驗證，最後撰寫[驗證報告](#驗證報告)儲存在**output_dir**內
- shell args 必帶參數
```
--task_name=$(TASK_NAME) \
--do_test=True \
--use_dev_dataset=$(USE_DEV_DATASET) \
--model_type=$(MODEL_TYPE) \
--data_dir=$(DATA_DIR) \
--output_dir=$(OUTPUT_DIR)
```

- python example
```python
import tensorflow as tf

output_dir = "checkpoints/sentiment/2/"
model = tf.keras.models.load_model(output_dir)
model.summary()
```
##### 驗證報告
- test_report.txt:儲存驗證資料的confusion matrix
```
        - report:      
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000        10
           1     0.9167    0.9167    0.9167        12
           2     0.9000    0.9000    0.9000        10

    accuracy                         0.9375        32
   macro avg     0.9389    0.9389    0.9389        32
weighted avg     0.9375    0.9375    0.9375        32

        - total_time:    217.657s
```

- test_result.tsv:儲存驗證資料的實際label和預測label比較圖
```
text,true,pred
噓:皇民是日本人的意思喔，罵窩們皇民是在稱讚吧,1,1
推:顏色好美！,0,0
...
過敏所以暫不考慮這間~~,1,1
推:一銀那麼好借怎麼都不借我,2,2
```

#### 展示模型
- 透過**output_dir**參數來指定模型位置，並進行**keras.load_model**的動作，再等待使用者輸入文章字串，最後輸出各種分類的分數，以及預測結果
- 可藉由console進行模型預測展示
- shell args 必帶參數
```
--task_name=$(TASK_NAME) \
--do_inference=True \
--model_type=$(MODEL_TYPE) \
--output_dir=$(OUTPUT_DIR)
```
- 使用範例
```
--> [input content]: 這家店超棒的
positive:0.9112719297409058, negative:0.027610205113887787, neutral:0.061117853969335556
predict result : positive

--> [input content]: 這家店超爛的
positive:0.27210354804992676, negative:0.6493704915046692, neutral:0.07852598279714584
predict result : negative

--> [input content]: 這家店普通
positive:0.3982468247413635, negative:0.13436941802501678, neutral:0.4673837423324585
predict result : neutral
```

### 混合訓練集
- 當訓練新模型時有設定到`load_model_dir`參數的話，會自動去創建新訓練集
- 當使用少量資料進行模型訓練時，需要加入些許的舊資料，來提升整體的準確率，這邊提供兩種混合的方式
    - 依照新資料集內各標籤的數量決定混合就資料的數目
        - *但如果新資料集內各標籤筆數大於3位數(1000以上)，則會選擇使用三個標籤的最小公倍數進行資料的填補*
        - eg: 假設新資料集各標籤筆數[0, 1, 489]，混合筆數[1000, 1000, 500] => [1000, 1001, 989]
    - 指定混合數量
        - eg: 假設新資料集各標籤筆數[0, 1, 489]，指定混合筆數100 => [100, 101, 589]
- click --help
```markdown
Usage: create_load_dataset.py [OPTIONS]

  Create new dataset, that have some old dataset

  Args:     
    maximum_digits:     old_dataset_name(str): old datasets file name.     
    new_dataset_name(str): new datasets file name.
    output_dataset_name(str): mix old and new datasets file name.
    mix_number(int): How many training sets to mix.
    minimum_content_length(int): Limit the minimum number of content when mix dataset.

Options:
  --old_dataset_name TEXT         舊資料集名稱  [default: train.tsv]
  --new_dataset_name TEXT         新資料集名稱  [default: new_train.tsv]
  --output_dataset_name TEXT      混合資料集名稱  [default: train.tsv]
  --mix_number INTEGER            混合多少數量的訓練集，如果沒設定，則是依照新資料集內的各標籤的數量進行舊資料的混合
                                  [default: 0]

  --minimum_content_length INTEGER
                                  混合資料時，舊資料抓取時的最少字數限制  [default: 20]
  --maximum_digits INTEGER        最大能接受資料筆數的位元個數，如果超過抓最小公倍數，沒超過則以10^maximum_di
                                  gits補資料，eg：1000筆 => 4位元，  [default: 3]

  --help                          Show this message and exit.
```

## 訓練工具使用步驟
- 預設使用`tf-hub`的model，是實驗過後正確率、效能最好的
1. 設定好Makefile檔案內變數，[Ref.](#情緒模型)
    - e.g: `DATA_DIR`、`OUTPUT_DIR`、`DEPLOY_DIR`、`LOAD_MODEL_DIR`...
2. 建立docker images
    - `make build_images`
3. 進行模型訓練
    - 訓練新模型
        - `make run_fine_tuning_sentiment`
    - 載入舊模型訓練新模型
        - `make run_load_model_to_train`
4. 若訓練結果不滿意
    1. 先進行模型驗證，來觀看模型預測與實際比對結果
        - `make run_test_sentiment`
    2. 手動調整有問題的訓練集or測試集，再重新訓練
    3. 若對結果還是不滿意，則可以考慮調整超參數
        - `learning_rate`、`num_epochs`、`batch_size`、`dropout_rate`...
5. 部屬模型
    - 先確定`serving/model/model.config`檔內的設定，可參考下面的[部屬說明](#部屬說明)
    - 設定Makefile的`SERVING_DIR`
    - `make deploy_model`，需注意`SERVING_DIR`內的模型版本號是否有衝突

## 部屬說明
- 當訓練好一個模型後並儲存成SavedModel格式後，我們透過TF serving進行模型的部屬，並用grpc進行呼叫
    - TF serving 簡單介紹
        - TF serving`支援同模組下多個版本的模型同時運行`(情緒模型下有1、2版)
        - TF serving`支援無痛上版、退版`，只需更改model.config即可
    - models.config 設定
        - models.config中使用`version_labels`(有key: string => 自定義的名稱、value: int => 對應的版本)來綁定不同版本對應到的名稱，以利於grpc呼叫時可呼叫不同版本
        - 需確認models.config中使用`versions`對應的`base_path`路徑下的資料夾名稱(即是版本號碼)是否一致
        
        <details>
        <summary>models.config設定範例</summary>
        
        ```
        model_config_list {
            config {
                name: 'sentiment',
                base_path: '/models/sentiment/',
                model_platform: "tensorflow",
                model_version_policy: {
                    specific: {
                        versions: 1,
                        versions: 2
                    }
                },
                version_labels: {
                    key: "stable",
                    value: 1
                },
                version_labels: {
                    key: "canary",
                    value: 2
                }
            }
        }
        ```
        </details>
        
- 部屬位置: `/serving/models/`
- 部屬設定檔: `/serving/models/models.config`
- 快速部屬: 如上面Quick start內的快速部屬
- 上版、退版(版本切換)流程:
    - 目前設定兩種版本
        1. 穩定版(stable)
        2. 測試版(canary)
    - 上版流程:
        - 假設原本版本為stable -> 1、canary -> 2，確認測試版(canary版本)穩定後可以上線時，即可修改stable -> 2，canary可不變。
    - 退版流程:
        - 假設原本版本為stable -> 2，今天上的版本有問題需要退版，則修改為stable -> 2 => 1，即可達成，不需再額外再進行設定。
- 參考網站: 
    - [TensorFlow serving](https://www.tensorflow.org/tfx/serving/serving_config#serving_multiple_versions_of_a_model)
    - [TensorFlow 2.x 模型 Serving 服務](https://www.mdeditor.tw/pl/pXfR/zh-tw)

## 實驗紀錄
- [更多的實驗記錄](https://gitting.eland.com.tw/rd2/models/sentiment-training-tool/-/wikis/home)

### 比較learning_rate對各模型的影響
- max_seq_length=128
- batch_size=32
- num_train_epochs=2
- predict_data=test.tsv(3506筆)
- train_data=train.tsv(14023筆)

- `bert-base-chinese` best test_acc: 0.9062
- `ckiplab/bert-base-chinese` est test_acc: 0.9016

| | lr |train_loss|train_acc|val_loss|val_acc|test_loss|test_acc|train_times|predict_times|
|---|---|---|---|---|---|---|---|---|---|
|bert-base-chinese(no train)|2e-5|1.4478|0.3872|1.4788|0.3782|1.4278|0.3889|207.03s|23.506s|
|bert-base-chinese(no val dataset)|2e-5|0.2531|0.9084| | |0.3169|0.9022|654.327s|23.53s|
|`bert-base-chinese`|2e-5|0.2491|0.9133|0.2656|0.9217|0.2989|`0.9062`|635.389s|22.797s|
|bert-base-chinese|5e-5|0.2360|0.9183|0.3150|0.8875|0.3095|0.8925|635.733s|22.816s|
|ckiplab/bert-base-chinese|2e-5|0.2517|0.9125|0.2811|0.9031|0.3176|0.8959|636.942s|22.86s|
|ckiplab/bert-base-chinese|5e-5|0.2664|0.9072|0.3025|0.9074|0.3568|0.8756|634.246s|22.808s|
|ckiplab/albert-base-chinese|2e-5|0.4138|0.8462|0.4197|0.8632|0.4093|0.8577|523.806s|19.81s|
|ckiplab/albert-base-chinese|5e-5|0.4277|0.8524|0.4144|0.8618|0.4012|0.8574|520.125s|19.839s|
|ckiplab/albert-tiny-chinese|2e-5|0.4922|0.8111|0.4460|0.8376|0.4463|0.8360|104.799s|3.375s|
|ckiplab/albert-tiny-chinese|5e-5|0.4301|0.8343|0.4130|0.8476|0.4326|0.8363|104.413s|3.363s|

### 比較各模型的架構
- TF1 model structure、TF2 tf-models-official model structure、TF2 transformers model structure
![model_structure](model_structure.png)

### 比較字詞長度與batch長度的關係
- 比較bert-base-chinese模型Seq-Length和Max-Batch-Size的關係
- 觀察多少的Seq-Length下，Max-Batch-Size等於多少不會OOM(Out-of-memory)
- max_seq_length : 別人trained好的模型是利用`512`長度進行訓練，但我們可以利用較小的長度進行fine-tuning來節省memory
- train_batch_size : train_batch_size跟memory是正比關係
- optimizer : BERT預設是Adam，但她需要額外的memory去存取`m`和`v`向量，我們可以選擇其他的optimizer來減少memory的使用，但這樣會引響結果，所以這個實驗不改變optimizer
- System : `bert-base-chinese`，因為這是我們上面實驗過Accuracy最高的模型

| Seq Length | Max Batch Size |
|------------|----------------|
| 64 | 64 |
| 128 | 32 |
| 256 | 16 |
| 320 | 14 |
| 384 | 12 |
| 512 | 6 |

### Siege_test
- 共同的請求數目，來觀測與原模型之間處理速度上的差異
- 顯卡: GeForce RTX 2070 super
- 請求次數: 5000
- h5 -> savedModel: 先儲存成h5來排除掉部分部屬不需要的資訊(opt、loss...)，再轉存成savedModel
- savedModel -> tensorRT: 從savedModel轉換成tensorRT，tensorRT儲存時其實還是savedModel，但修改了內部graph架構

|   | Elap Time | Data Trans | Resp Time | Trans Rate | Throughput | Concurrent | OKAY | Failed |
|---|-----------|------------|-----------|------------|------------|------------|------|--------|
| 原模型 | `268.65` | 1 | 0.11 | 18.61 | 0.00 | 2.00 | 5000 | 0 |
| savedModel(no_train) | 291.28 | 1 | 0.12 | 17.17 | 0.00 | 2.00 | 5000 | 0 |
| savedModel | 377.44 | 1 | 0.15 | 13.25 | 0.00 | 2.00 | 5000 | 0 |
| savedModel -> tensorRT | 358.67 | 1 | 0.14 | 13.94 | 0.00 | 2.00 | 5000 | 0 |
| h5 -> savedModel | 375.54 | 1 | 0.15 | 13.31 | 0.00 | 2.00 | 5000 | 0 |
| h5 -> savedModel -> tensorRT | 361.04 | 1 |0.14 | 13.85 | 0.00 | 2.00 | 5000 | 0 |
| tensorRT_fp16_n15_c100_tensorRT| 371.09 | 1 | 0.15 | 13.47 | 0.00 | 2.00 | 5000 | 0 |
| ckiplab_bert-savedmodel | 381.34 | 1 | 0.15 | 13.11 | 0.00 | 2.00 | 5000 | 0 | 
| `tensorflow-hub-h5 -> savedmodel` | `291.66` | 1 | 0.12 | 17.14 | 0.00 | 2.00 | 5000 | 0 |
| `tensorflow-hub-h5 -> savedmodel -> tensorRT` | `291.63` | 1 | 0.12 | 17.15 | 0.00 | 2.00 | 5000 | 0 |

### 模型容量比較
- 相同的訓練集，來比較哪種儲存類型所需的`.pd`、`variables`較小

|   | saved_model.pb | variables.data-00000-of-00001 | model.tflite |
|---|----------------|-------------------------------|--------------|
| 原模型 | 1020K | 390M | - |
| savedModel | 10M | 1.1G | - |
| h5 -> savedModel | 9.5M | 390M | - |
| tflite -> tf serving | - | - | 99M |
| savedmodel -> tensorRT -> tensorRT | 790M | 1.1G | - |
| h5 -> savedmodel -> tensorRT | 401M | 390M | - |
| `Tensorflow-hub h5 -> savedmodel` | 14M | 393M | - |
| `Tensorflow-hub h5 -> savedmodel -> tensorRT` | 795M | 393M | - |

## 參考資料
- https://www.tensorflow.org/official_models/fine_tuning_bert
- https://huggingface.co/transformers/master/model_doc/bert.html#tfbertforsequenceclassification
- https://github.com/ckiplab/ckip-transformers
- https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html
- https://arxiv.org/pdf/1706.03762.pdf

## FAQ

**模型儲存**

- How to save the whole model as SavedModel format for inference?
    - [answer](https://github.com/huggingface/transformers/issues/6864#issuecomment-690086932)

**模型部屬**

- 可以用`nvtop`進行觀測，看顯卡的Memory有沒有吃到
    - Memory不夠，導致模型讀不進來，調整bin/run_serving.sh內的docker設定`CONTAINER_RAM`、`CONTAINER_CPU`可以提高為`8g`

## Future_work
- 加入標題進行訓練，藉此增加輸入特徵，來提高短文的預測能力
    - [討論] Youtube要移除所有質疑選舉結果的影片；`推: 人民幣真香`
    - [討論] Youtube要移除所有質疑選舉結果的影片；推: 這也太噁
    - [討論] Youtube要移除所有質疑選舉結果的影片；推: 這麼厲害
