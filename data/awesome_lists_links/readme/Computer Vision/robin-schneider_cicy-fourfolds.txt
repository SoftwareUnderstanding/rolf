# cicy-fourfolds

This repository contains code to learn the Hodge number of Complete Intersection Calabi-Yau manifolds. The scripts work for three- and four-folds. The accompanying paper can be found [here (2108.02221)](https://arxiv.org/abs/2108.02221).

## Set-up

There should be no package conflict when making a new conda environment and installing packages with

```console
conda create -n fourfolds python=3.7
conda activate fourfolds
pip install -r requirements.txt
```

If you are interested in learning four-folds you will however need a modern GPU and should check out tensorflow gpu installation [instructions](https://www.tensorflow.org/install/gpu).

Once successfully installed you can prepare the CICY four-fold dataset with

```console
python create_data.py
```

and run a [hyperparameter optimization](https://github.com/automl/HpBandSter) with

```console
python BOHB.py --min_budget=10 --max_budget=100 --nfold=4 --shared_directory=bohb4 --hodge=2
```

this will usually take a couple of weeks on a modern desktop. Alternatively you can use the best classification parameters we found with

```console
python four_fold_worker.py
```

The plots in the paper were made with a different set of hyperparameters which balanced accuracy vs model parameters in a more reasonable way. They can be accessed with

```console
python four_fold_worker.py --fconfig=efficient.in
```

Note: the hyperparameters found for a classification run do generically not generalize well to a regression architecture. You will have to do a separate hyperparameter optimization for the regression case. It is also possible to predict all four hodge numbers at the same time:

```console
python BOHB.py --min_budget=10 --max_budget=100 --nfold=4 --shared_directory=bohb4 --hodge=-1 --classification=0
```

Such a scan will take even more time and the results are not comparable to the accuracy achieved by the [CICYminer](https://github.com/thesfinox/ml-cicy-4folds).
The reasons for that are as follows: The current implementation only shares the Inception modules but the different tasks do not have their own Inception blocks. It also does not utilize the Huber Loss. Training with additional aux-losses is implemented, but increases computation time significantly.

## Notebooks

There are two notebooks

1. The [first](/exploring_fourfolds.ipynb) notebook explores CICY four-folds and generates the histogram plots.
2. The [second](/accuracy_plots.ipynb) notebook plots the results found from the 'efficient' run which will give you the following pretty picture.
![accuracies](./plots/accuracies.png "Train:Val:Test accuracies.")

## Questions

are welcome.

## References and links

This project is a follow up to [previously](https://github.com/thesfinox/ml-cicy) initiated [studies](https://arxiv.org/pdf/2007.13379.pdf) investigating CICY three-folds using [inception](https://arxiv.org/abs/1409.4842) inspired neural networks. The [CICYminer](https://github.com/thesfinox/ml-cicy-4folds) used in the main results section is based on the [Deep Miner](https://arxiv.org/abs/2102.09321) architecture for image recognition.