# CCA for Visual Dialogue

We apply the classical statistical method of Canonical Correlation Analysis (CCA) [[Hotelling, 1926](https://academic.oup.com/biomet/article/28/3-4/321/220073); [Kettenring, 1971](https://www.jstor.org/stable/2334380?seq=1#metadata_info_tab_contents)] to the task of visual dialogue - a sequential question-answering task where the questions and
answers are related to an image. With CCA, we learn mappings for questions and answers (and images) to a joint embedding space, within which we measure correlation between test set questions and their corresponding candidate answers in order to rank the answers. We show comparable performance in mean rank (MR), one of the established metrics on the [Visual Dialogue](http://www.visualdialog.org) dataset, to state-of-the-art models which have at least an order of magnitude more parameters. We use this surprising result to comment on potential issues regarding the current formulation and evaluation of the visual dialogue task.

Our *NeurIPS 2018 Critiquing and Correcting Trends in Machine Learning* workshop paper can be found [here](http://arxiv.org/abs/1812.06417)

## Install dependencies

This project is implemented in [PyTorch](http://www.pytorch.org). It is recommended to create an Anaconda environment for the project and all associated package dependencies:
```
conda create -n cca_visdial python=3.6 pip
source activate cca_visdial
conda install pytorch torchvision cuda80 -c pytorch
bash install_deps.sh
```

The project was built using Python 3.6, PyTorch 1.0 and CUDA 8.0. Check [PyTorch](http://www.pytorch.org) for other versions and operating systems.

## Download datasets

The project uses the [Visual Dialog](http://www.visualdialog.org) dataset. To download and prepare the necessary datasets, run:
```
bash download_and_prepare.sh -d <data_dir> # if left unspecified, datasets are downloaded to ./data
```
This will download [Microsoft COCO](http://www.mscoco.org/dataset) `train2014` and `val2014` images as well as *v0.9* and *v1.0* of the Visual Dialog dataset (images and dialogues).

## CCA for Answer-Question and Answer-Image-Question

### Ranking candidate answers

To run the CCA algorithm on the Visual Dialogue *v0.9* dataset with default settings, use the following command:
```
python src/main.py --datasetdir <your_dataset_dir> --results_dir <your_results_dir> --gpu <0-indexed gpu id> 
```
where `your_dataset_dir` and `your_results_dir` defaults to `/data` and `/results` in the root folder respectively. 

This computes average [FastText](https://fasttext.cc) vectors for the questions and answers, and then applies two-view CCA on the (train) representations to obtain a pair of projection matrices which maximises their correlation in the joint embedding space. Using the learned matrices, the test questions and their corresponding candidate answers are projected into the space, and the cosine distance between the projections is used to rank the candidates. By observing the position of the assigned ground-truth answer in the ranked list, the mean rank (MR), mean reciprocal rank (MRR) and recall@{1, 5, 10} are computed across the dataset.

It is also possible to run three-view CCA on the questions, answers *and* images (represented by their [ResNet34](https://arxiv.org/abs/1512.03385) features) using `--input_vars answer --condition_vars img_question`. Here, the projected answers and *questions* (rather than images) are used to rank the candidate answers.

To *view* the ranked candidates answers for given questions, use `--interactive --batch_size 1`. This will print out the ranked candidate answers for an image and its
associated 10 questions.

### Generating on-the-fly candidate answers

The above computes the MR, MRR and recall@{1,5,10} using the candidate answer sets provided in the [Visual Dialogue](http://www.visualdialog.org) dataset. It is also possible,
however, to construct candidate answer sets on-the-fly using CCA: the closest questions to the test question are drawn from the training set, and their corresponding answers
extracted. The ranking metrics (MR, MRR and recall) cannot be computed in this case since the labelled ground-truth answer is no longer valid. Use the `--on_the_fly k`
flag to construct a set of k candidates, and, as before, `--interative --batch_size 1` to qualitatively view the ranked (by correlation) on-the-fly candidate answers for a given image and its questions.

### Analysing top-ranked answers

We quantify the validity of the top-ranked answers from the *VisDial* candidates in relation to the ground truth using a heuristic based on their correlations:

For any given question and candidate answer set, we cluster the answers
based on an automatic binary thresholding ([Otsu (1979)](https://ieeexplore.ieee.org/document/4310076)) of their
correlations with the given question. We then compute:
(1) The average standard deviation of the correlations in the lower-ranked split,
(2) The number of answers (out of 100) falling in the lower-ranked split, and
(2) The fraction of questions whose correlation with the ground truth answer is higher than the threshold.

This quantifies (1) how closely clustered the top answers are, (2) how large the set of highly correlated answers is, and (3) how often the
ground-truth answer is in this cluster, respectively. Low values for the first, and high values for the second and third
would indicate that there exists an equivalence class of answers, all relatively close to the ground-truth
answer in terms of their ability to answer the question.

To compute statistics (1), (2), and (3), use the `--threshold` flag.

### Uploading to EvalAI Visual Dialog Challenge 2018 Server

Results on Visual Dialogue *v1.0* can additionally be uploaded to the [EvalAI Visual Dialog Challenge Evaluation Server](https://evalai.cloudcv.org/auth/login). For the validation set, the standard MR, MRR and
recall@{1,5,10} will be computed. For the test set, an additional metric, the normalised discounted cumulative gain (NDCG) will be computed. 

To specify this option, use `--datasetversion 1.0 --batch_size 1 --evalset <val or test> --save_ranks`

## Citation

```
@article{massiceti2018visual,
  title={Visual Dialogue without Vision or Dialogue},
  author={Massiceti, Daniela and Dokania, Puneet K and Siddharth, N and Torr, Philip HS},
  journal={arXiv preprint arXiv:1812.06417},
  year={2018}
}
```
