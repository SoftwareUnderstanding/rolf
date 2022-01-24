# Image-Classfication-Tasks


### **Week 1**
- **Learning Objectives:** Basic image classification tasks with simple models using **Pytorch** and **Jax**
- **Learning Outcomes:** Finish using Pytorch and Jax to build a Lenet-5 model to do image classification tasks.

- **Findings & Conclusion:**

| Framework/Diff     | Model Bulding     | Model Training     | Others |
| ---------- | :-----------:  | :-----------: | :-----------: |
| Pytorch    | - Model is a class <br> - Define each layer as variables  <br> - Foward Function manually goes through each layer   | - Has dataloader that can be enumerate <br> - Usually use built-in loss function <br>  - Usually we call **optimizer.step()** to update   | - Provide plenty of datasets |
| Jax    | - Model is treated as a function <br> - **stax** for example returns model parameters and foward function     | - Has to define data_stream method <br>   - Has to self-define loss function <br> - Update each batch state and pass to the next   | - Has to manually define dataset |


### **Week 2**
- **Learning Objectives:** Getting started on Julia and corresponding libraries.
- **Material**: 
- - https://syl1.gitbook.io/julia-language-a-concise-tutorial/language-core/getting-started
- - https://github.com/denizyuret/Knet.jl/tree/master/tutorial
- **Learning Outcomes:** Finish using Julia and Knet building Lenet5 and classify MNIST data.
- **Findings & Conclusion:**
- - Julia is very similar to Python
- - Package Reference is not very clear
- - It can be very easy and clear to build each layer of NN by using the Julia Constructor

### **Week 3**
- **Learning Objectives:** Try to load self-defined data to each of the three previous learned pipeline.
- **Learning Outcomes:** Finish using Julia, Pytorch, and Knet to load custome dataset.

### **Week 4-10: Classify the X-Ray dataset using different models w/ high accuracies**
- **Outline:** 
<br />The NIH Chest X-rays dataset is composed of about 110,000 chest X-ray images with 15 classes (14 diseases, and one for "No findings"). We are going to build/utilize different models to perform classification.
- **Dataset & Preparation:**
<br />For loading data only, Zipfile is an easy way to load data. You probably do not want to unzip the whole dataset (~90G) if you do not plan to train.
The code below will help you orient all the data paths and, assuming you want to do binary classification, truning lables into binary.
```python
 zf = z.ZipFile(data_path) 
 df = pd.read_csv(zf.open('Data_Entry_2017.csv')) # load paths&labels
 
 img_name = df.iloc[1, 0]
 df = df.loc[:, "Image Index":"Finding Labels"]

 # Data Preparation
 img_paths = {os.path.basename(name): name for name in zf.namelist() if name.endswith('.png')}
 df['path'] = df['Image Index'].map(img_paths.get)
 df.drop(['Image Index'], axis=1,inplace = True) # keep path and labels only

 # Make the data binary
 labels = df.loc[:,"Finding Labels"]
 one_hot = []
 for i in labels:
    if i == "No Finding":
         one_hot.append(0)
    else:
         one_hot.append(1)
 one_hot_series = pd.Series(one_hot)
 one_hot_series.value_counts()
 df['label'] = pd.Series(one_hot_series, index=df.index)
 df.drop(['Finding Labels'], axis=1,inplace = True)
```
If you print the data frame, you should see something like this:
```r
                                 path  label
0  images_001/images/00000001_000.png      1
1  images_001/images/00000001_001.png      1
2  images_001/images/00000001_002.png      1
3  images_001/images/00000002_000.png      0
4  images_001/images/00000003_000.png      1
```
- **Experiment 1:**
  - Model: Pytorch pretrained Resnet50
  - Trainable Layers: Layer2-4 and FC
  - Number of Classes: 2(Positive or Negative)
  - Best Achieved Accuracy: .93
- **Experiment 2:**
  - Model: ChexNet
  - Trainable Layers: All
  - Number of Classes: 15 (14 Diseases and 1 No Finding)
  - Best Achieved Accuracy: .8
- **Next Step:**
  - Ensemble
  - Hierarchical-Learning
  - Label-smoothing
- **Reference & Resources:**
  <br /> https://www.kaggle.com/nih-chest-xrays/data 
  <br /> https://jrzech.medium.com/reproducing-chexnet-with-pytorch-695ff9c3bf66
  <br /> https://arxiv.org/pdf/1711.05225.pdf
  <br /> https://stanfordmlgroup.github.io/competitions/chexpert/
