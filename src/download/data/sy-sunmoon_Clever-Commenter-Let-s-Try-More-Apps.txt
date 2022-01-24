# Clever Commenter: Let's Try More Apps
This repo contrains of the **Clever Commenter: Let's Try More Apps** project in Google AI ML Winter Camp.

## What Problem We solve
Comments are one of the most important ways for App downloaders to understand this App. However, many newly released (online) Apps have few comments, which seriously affects the user's interest and enthusiasm of those apps. Therefore, in order to **help App downloaders better understand the newly released Apps**, we designed an automatic comment generator called "**Clever Commenter: Let's Try More Apps**".

## What is "Clever Commenter: Let's Try More Apps"
**Clever Commenter: Let's Try More Apps** is an interesting and powerful automatic comment generator. It consists of the following modules:
- **Key-words Extraction**: This module uses the structure data of the app (such as `Category`, `Age group`, `Price`) to find the most relevant apps based on Social Network theory instead of basic low order similarity. Then extracts the key-words of the related apps as an alternative of the newly released App. 
- **Key-words Based Review Generator**: This module generates a review based on given key-words. Key-words are extracted by the first module or input from the App designers.
- **Review Sentiment Transfer**: This module transfer a negative review into a positive review, and vice versa. In this way, "Clever Commenter: Let's Try More Apps" can control the emotion of the generated reviews.

****************************************

### Module1: Key-words Extraction
The model aims to find APP's most similar APPs based on Social Network theory instead of basic low level similarity, then extract these APP's keywords.

#### 1. Dataset
In our example, we use [Google Play Store Apps Dataset](https://www.kaggle.com/lava18/google-play-store-apps#googleplaystore_user_reviews.csv) as our source data.

#### 2. keywords extraction model
By run the follwing files, go to the keywords-extraction folder, and you can get each APP's most similar APP's keywords.

- <code>get_ppmi_matrix.py</code> can calculate each existing APP's high level similarity , by the Soical Network theory Random Walk.

- <code>loworder_similarity_to_highorder_similarity_model.py</code> can train and predict APP's high level similarity with other existing APPs.

- <code>change_ppmi_matrix_to_similar_app.py</code> can get each APP's most similar APP's name.

- <code>convert_orl_data_to_keyword_by_Category.py</code> can get each category APPs' top non-emotional keywords and emotional keywords.

- <code>convert_orl_data_to_keyword_of_each_app_by_similar_app.py</code> can get each APP's most similar APPs' top non-emotional keywords and emotional keywords.

****************************************

### Module2: Key-words Based Review Generator
The model aims to generate fluent and reasonable reviews based on the input keywords describing the product.

#### 1. Data Preprocess
Before running the review-generator/preprocess.py, your should provide the following files in the <code>data/source_data/</code> folder:

- <code>XX.src1</code> is the file of the input keywords.
- <code>XX.src2</code> is the file of the concepts extracted from [ConceptNet](http://conceptnet.io/). 
- <code>XX.tgt</code> is the file of the output reviews.

Run preprocess.py as following, and the preprocessed files are stored in the <code>data/save_data/</code> folder.
```bash
python3 preprocess.py --load_data data/source_data/ --save_data data/save_data/
```

#### 2. Train
To train a model, go to the review-generator folder and run the following command:
```bash
python3 train.py --gpus gpu_id --config config.yaml --log log_name 
```

#### 3. Test
To test the well-trained model, go to the review-generator folder and run the following command:
```bash
python3 predict.py --gpus gpu_id --config config.yaml --restore checkpoint_path --log log_name 
```

****************************************

### Module3: Review Sentiment Transfer

The model learns to transfer a negative sentiment review into a positive one without any parallel data.

#### 1. Data Preprocess
After running the sentiment-transfer/format_data.py, it can generate three files in the <code>sentiment_transfer</code> folder:

<code>train.0</code>, <code>dev.0</code>, <code>test.0</code> denotes the negative train/dev/test files

<code>train.1</code>, <code>dev.1</code>, <code>test.1</code> denotes the positive train/dev/test files
<br>

#### 2. Train

To train a model, go to the sentiment-transfer folder and run the following command:
```bash
python style_transfer.py --train ../data/sentiment_transfer/train --dev ../data/sentiment_transfer/dev --output ../tmp/sentiment.dev --vocab ../tmp/google.vocab --model ../tmp/model
```

#### 3. Test

- ##### Test file has sentiment labels
If the test file has sentiment labels, just run the following command:
```bash
python style_transfer.py --test ../data/sentiment_transfer/test --output ../tmp/sentiment_transfer.test --vocab ../tmp/google.vocab --model ../tmp/model --load_model true
```

- ##### Test file doesn't have sentiment labels
If the test file doesn't have sentiment labels, such as the generated reviews, just run the following model to train a binary sentiment classifier. And then load the trained model to detect which generated review is negative or positive.
```bash
# train
python classifier.py --train ../data/sentiment_transfer/train --dev ../data/sentiment_transfer/dev --vocab ../tmp/google.vocab --model ../tmp/classifer-model 
# test
python classifier.py --test TEST_FILE_PATH --output OUTPUT_FILE_PATH --vocab ../tmp/google.vocab --model ../tmp/model --load_model true
```
And then, run the follow code to get the transferred review:
```bash
python style_transfer.py --test OUTPUT_FILE_PATH --output ../tmp/sentiment_transfer.test --vocab ../tmp/google.vocab --model ../tmp/model --load_model true
```

***************************************************************
## Web to show the demo 
1. Run the http server to allow the js. script.
```bash
python3 -m web/run_server.sh &
```
2. Visit web/demo.htm to watch the demo.

***************************************************************

## Cite
This code is based on the following paper:

<i> "deep neural network for learning graph representations". Cao, Shaosheng . Thirtieth Aaai Conference on Artificial Intelligence AAAI Press, 2016.[aaai.org](https://aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12423/11715) </i>

<i> "Style Transfer from Non-Parallel Text by Cross-Alignment". Tianxiao Shen, Tao Lei, Regina Barzilay, and Tommi Jaakkola. NIPS 2017. [arXiv](https://arxiv.org/abs/1705.09655) </i>

<i> "End-To-End Memory Networks". Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus. NIPS 2015. [arXiv](https://arxiv.org/abs/1503.08895) </i>

<i> "Dynamic Memory Networks for Visual and Textual Question Answering". Caiming Xiong, Stephen Merity, Richard Socher. 2017. [arXiv](https://arxiv.org/abs/1603.01417) </i>

## Author
Yue Sun (孙悦）, WeiTu（涂威）, [FuliLuo](https://scholar.google.com/citations?user=1s79Z5cAAAAJ&hl=zh-CN)（罗福莉）
