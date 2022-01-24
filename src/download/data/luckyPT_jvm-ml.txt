# 简介
基于java，scala，spark的机器学习相关算法

### 相较于python的优势
1. 大部分线上服务以java为主，因此模型服务与现有的服务结合比较方便<br>
2. 可以利用spark处理大规模数据，spark对scala的支持优于python

# DEMO 目录
- [逻辑回归](/src/main/java/com/pt/ml/algorithm/LogisticRegression.scala)

- [最大熵模型及自定义实现](/src/main/java/com/pt/ml/algorithm/MaxEntropy.java)

- [决策树分类及调优总结(CART树)](/src/main/java/com/pt/ml/algorithm/DecisionTree.scala)

- [spark GBDT算法](/src/main/java/com/pt/ml/algorithm/GradientBoostTree.scala)

- [spark 随机森林算法](/src/main/java/com/pt/ml/algorithm/RandomForest.scala)

- [线性支撑向量机](/src/main/java/com/pt/ml/algorithm/SurportVectorMerchine.scala)

- [xgboost for scala](/src/main/java/com/pt/ml/algorithm/Xgboost4jScala.scala)

- [xgboost for spark](/src/main/java/com/pt/ml/algorithm/Xgboost4jSpark.scala)

- [LDA 聚类](/src/main/java/com/pt/ml/algorithm/LdaCluster.scala)

- [spark生成tfRecord文件以及python读取的demo](/src/main/java/com/pt/ml/generate/tfrecord/GenerateTfrecordBySpark.scala)

- [二分类的ROC曲线、PR曲线、阈值与PR曲线、阈值F1曲线绘制](/src/main/java/com/pt/ml/util/BinaryClassEvaluation.scala)

- [多分类评估指标计算,精确度、加权准确率召回率、F1值、各类别的准确率-召回率-F1值](/src/main/java/com/pt/ml/util/MultiClassEvaluation.scala)

- [Ansj分词](/src/main/java/com/pt/ml/process/AnsjSegmenterUtil.java)

- [JieBa分词](/src/main/java/com/pt/ml/process/JiebaSegmenterUtil.java)

- [自定义简易分词](/src/main/java/com/pt/ml/process/CustomSegmenter.java)

- [数据标准化与归一化](/src/main/java/com/pt/ml/process/Scaler.scala)

- [TF-IDF计算](/src/main/java/com/pt/ml/process/TfIdf.scala)

- [生成词向量](/src/main/java/com/pt/ml/process/WordToVector.scala)

- [连续特征离散化](/src/main/java/com/pt/ml/process/Discretization.scala)

- [计算编辑距离](/src/main/java/com/pt/ml/example/EditDistanceDemo.java)

- [SimHash算法](/src/main/java/com/pt/ml/process/SimHash.java)

- [PCA降维](/src/main/java/com/pt/ml/process/Pca.scala)

- [TSNE降维](/src/main/java/com/pt/ml/process/TSNEStandard.java)

- [fastText for java训练](/src/main/java/com/pt/ml/algorithm/FastText4J.scala)

- [fastText for java 词向量模型使用](/src/main/java/com/pt/ml/deeplearning/nlp/Word2VecFastText.java)

- [java 绘制点、线、柱状图](/src/main/java/com/pt/ml/visualization)

- [deeplearning4j - 单机前向传播神经网络](/src/main/java/com/pt/ml/deeplearning/BpNeuralNetwork.java)

- [deeplearning4j - 单机卷积神经网络](/src/main/java/com/pt/ml/deeplearning/CnnNeuralNetwork.java)

- [deeplearning4j - spark版本卷积神经网络](/src/main/java/com/pt/ml/deeplearning/CnnNeuralNetworkSpark.scala)

- [deeplearning4j - 单机循环神经网络](/src/main/java/com/pt/ml/deeplearning/LstmClassification.java)

- [deeplearning4j - spark版本循环神经网络](/src/main/java/com/pt/ml/deeplearning/LstmClassificationSpark.scala)

- [deeplearning4j - 词向量使用](/src/main/java/com/pt/ml/deeplearning/nlp/Word2VecDeeplearning4j.java)


# 开源引用

## License
BSD

## References
(From fastText's [references](https://github.com/facebookresearch/fastText#references))

Please cite [1](#enriching-word-vectors-with-subword-information) if using this code for learning word representations or [2](#bag-of-tricks-for-efficient-text-classification) if using for text classification.

### Enriching Word Vectors with Subword Information

[1] P. Bojanowski\*, E. Grave\*, A. Joulin, T. Mikolov, [*Enriching Word Vectors with Subword Information*](https://arxiv.org/abs/1607.04606)

```
@article{bojanowski2016enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.04606},
  year={2016}
}
```

### Bag of Tricks for Efficient Text Classification

[2] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/abs/1607.01759)

```
@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}
```

### FastText.zip: Compressing text classification models

[3] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, [*FastText.zip: Compressing text classification models*](https://arxiv.org/abs/1612.03651)

```
@article{joulin2016fasttext,
  title={FastText.zip: Compressing text classification models},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1612.03651},
  year={2016}
}
```

(\* These authors contributed equally.)
