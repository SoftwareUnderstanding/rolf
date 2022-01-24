# Deep Learning assignments (DL-MAI)

This is the repository containing the source code for the assignments of the Deep Learning course at the Master in Artificial Intelligence at UPC-BarcelonaTech. The code is based on PyTorch. Authors:

    - Jordi Armengol
    - Albert Rial

## Requirements

Provided Python3.7 and CUDA are already installed in the system, run:

```
bash setup.sh
```

This script activates the virtual environment, downloads the required dependencies and marks 'src' as the sources root.

For obtaining the data of the CNNs and transfer learning assignments (mit67), run:

```
bash get-mit67.sh
python src/cnn/preprocess.py

```

In the case of RNNs (DeepMind's Mathematics Dataset):

```
get-deepmind-mathematics.sh
```

## Structure
    - data/: Directory where data will be downmloaded.
    - experiments/: Directory where experiments are stored.
    - src/: Source code:
        - cnn/: Code for CNNs (mit67).
        - rnn/: Code for RNNs (DeepMind's mathematic reasoning dataset).
        - transfer/: Code for transfer learning (mit67).

Each sub-directory within src/ has (apart from other components):
   - train.py: Script to train a given model (with a given configuration) on a given dataset.
   - evaluate.py: Script to evaluate a given model on a given dataset.
   - create_experiment.py: Utility to create an experiment with a given configuration (Slurm, Bash).

## Convolutional Neural Networks

### Instructions

Preprocessing:

```
python src/cnn/preprocess.py

```
In the preprocessing script, we:
    
    - Print some statistics of the dataset and the preprocessing procedure.
    - Remove a few malformatted images.
    - Remove a few BW images.
    - Convert a few PNG into JPG.
    - Perform a stratified split into train, validationn and test (80-10-10).
    - By default, we resize all images to 256x256, even if our model is input-size-agnostic.

Creating experiment:

```
python src/cnn/create_experiment.py experiment_name [options]

```
This script generates a new directory for the experiment in the experiments/ directory. For each experiment, bash, 
batch (Windows) and Slurm scripts are generated, and they must be executed from the corresponding experiment directory.
The text logs (.out, .err and .log) files, as well as the Tensorboard logs, will always be stored in the directory of
the corresponding experiment.

For directly launching the training (without any training), even if that is not recommended, run:
```
python src/cnn/train.py [options]

```

The training options are the following (note that the defaults have been optimized for performance in the validation set):
```
parser = argparse.ArgumentParser(description='Train a CNN for mit67')
parser.add_argument('--arch', type=str, help='Architecture', default='PyramidCNN')
parser.add_argument('--data', type=str, help='Dataset', default='256x256-split')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
parser.add_argument('--lr', type=float, help='Learning Rate', default=0.0001)
parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
parser.add_argument('--no-augment', action='store_true', help='disables data augmentation')
parser.add_argument('--optimizer', type=str, help='Optimizer', default='Adam')
parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=32)
parser.add_argument('--criterion', type=str, help='Criterion', default='label-smooth')
parser.add_argument('--smooth-criterion', type=float, help='Smoothness for label-smoothing', default=0.1)
parser.add_argument('--early-stop', type=int,
                    help='Patience in early stop in validation set (-1 -> no early stop)', default=6)
parser.add_argument('--weight-decay', type=float, help='Weight decay', default=0.001)

parser.add_argument('--kernel-size', type=int, help='Kernel size', default=3)
parser.add_argument('--dropout', type=float, help='Dropout in FC layers', default=0.25)
parser.add_argument('--no-batch-norm', action='store_true', help='disables batch normalization')
parser.add_argument('--conv-layers', type=int, help='N convolutional layers in each block', default=2)
parser.add_argument('--conv-blocks', type=int, help='N convolutional blocks', default=5)
parser.add_argument('--fc-layers', type=int, help='N fully-connected layers', default=2)
parser.add_argument('--initial-channels', type=int, help='Channels out in first convolutional layer', default=16)
parser.add_argument('--no-pool', action='store_true', help='Replace pooling by stride = 2')

parser.add_argument('--autoencoder', action='store_true', help='Train autoencoder instead of classification')

```

The default options are the ones that we observed to perfom better in our experiments.

Notice that the ```--autoencoder``` option allows to unsupervisedly pre-train an autoencoder. For doing so, launch an
experiment with the desired configuration and the ```--autoencoder``` option. Then, move the generated
```checkpoint_best.pt``` to the experiment directory, change its name to ```checkpoint_last.pt``` and start the training
procedure with the same architecture options, but without the ```--autoencoder``` option. The training script will
automatically start the training from the pre-trained encoder instead of from scratch.

For evaluating a model:
```
python src/cnn/evaluate.py [options]

```
The evaluation options are the following (notice that ensembles are implemented by providing more than one checkoint):
```
parser.add_argument('--arch', type=str, help='Architecture')
parser.add_argument('--models-path', type=str, help='Path to model directory',nargs='+')
parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pt',  help='Checkpoint name')
parser.add_argument('--data', type=str, help='Dataset', default='256x256-split')
parser.add_argument('--subset', type=str, help='Data subset', default='test')
parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=2)
```

### Experiments highlights

We conducted as many as 150 experiments in the CTE-POWER cluster (V100 GPUs) at the Barcelona Supercomputing Center.

We used the mit67 dataset (<http://web.mit.edu/torralba/www/indoor.html>), a well-known indoor scene recognition. It is
challening because it only contains 15,620 images, and by manually inspecting the 67 (imbalanced) classes, we can
observe that they are not easily distinguishable. Notice that for this assignment we were not supposed to use transfer
learning from another dataset (eg. VGG16), which made the problem more difficult.

By evaluating the different configurations in the validation set and comparing the results, we highlight the following
conclusions:
    
    - Autoencoder pre-training did not lead to better results.
    - Since the dataset is tiny, data augmentation was key to improve our results.
    - After performing a grid search on a number of architectural hyperparameters, we found a model with 5 convolutional
    blocks, with 2 convolutional layers per block with a kernel size of 3, and 2 fully-connected layers to be the best
    performant model.
    - Our models starts with 16 channels and doubles the number of channels for each block, while the image size is
    divided by 2 with pooling. This is why we call the architecture 'PyramidCNN'. Both 32 and 8 initial channels,
    performed worse, leading to over and underfitting, respectively.
    - Using stride instead of pooling did not lead to any improvement (actually, it performed worse).
    - Batch normalization remarkably improved the results.
    - For regularizing, we found that a dropout (only in the convolutional layers) of 0.25 and a weight decay of 0.001
    were the best options. We had to add some patience to the early stopping procedure. A label smoothing of 0.1 was
    also very useful.
    - Ensembling independently trained models reulted in a gain of about 8 points in accuracy.
    - Since the images present in the dataset have global features, we tried to incorporate Non-Local blocks
    (<https://arxiv.org/abs/1711.07971>), but we did not observe any gains.

The best found configuration, corresponding to the default values in the ```train.py``` script, obtained an accuracy of  58  in the validation set. Then, we build an ensemble of 10 independently trained classifiers with the same configuration, and obtained a validation accuracy of 64. We selected this ensemble as our final model, and obtained a test accuracy of 63.

## Recurrent Neural Networks
The goal of this work is to build an end-to-end neural system to answer mathematical questioNS. We will do so with a vanilla, character-level Seq2seq. 
### Instructions

We sub-sample a subset from the train-easy subset.
```
bash sample-mathematics.sh
```

We do not apply any preprocessing of the data (apart from uncasing). We provide raw characters to the neural network.

Creating experiment:

```
python src/RNN/create_experiment.py experiment_name [options]

```
This script generates a new directory for the experiment in the experiments/ directory. For each experiment, bash, 
batch (Windows) and Slurm scripts are generated, and they must be executed from the corresponding experiment directory.
The text logs (.out, .err and .log) files, as well as the Tensorboard logs, will always be stored in the directory of
the corresponding experiment.

For directly launching the training (without any training), even if that is not recommended, run:
```
python src/RNN/train.py [options]

```

The training options are the following (note that the defaults have been optimized for performance in the validation set):

```
parser = argparse.ArgumentParser(description="Train a RNN for Deepmind's Mathematics Dataset")
    parser.add_argument('--arch', type=str, help='Architecture', default='elman')
    parser.add_argument('--problem-types', type=str, nargs='*', help='List of problems to load from dataset',
                        default = ['numbers__base_conversion.txt', 'numbers__div_remainder.txt', 'numbers__gcd.txt',
                                   'numbers__is_factor.txt', 'numbers__is_prime.txt', 'numbers__lcm.txt',
                                   'numbers__list_prime_factors.txt', 'numbers__place_value.txt', 'numbers__round_number.txt'])
    parser.add_argument('--dataset-instances', type=int, default=100000,
                        help='Number of total instances we want to load from the dataset')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='Learning Rate', default=0.001)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--optimizer', type=str, help='Optimizer', default='Adam')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=64)
    parser.add_argument('--criterion', type=str, help='Criterion', default='xent')
    parser.add_argument('--smooth-criterion', type=float, help='Smoothness for label-smoothing', default=0.1)
    parser.add_argument('--early-stop', type=int,
                        help='Patience in early stop in validation set (-1 -> no early stop)', default=10)
    parser.add_argument('--weight-decay', type=float, help='Weight decay', default=0.0001)
    parser.add_argument('--dropout', type=float, help='Dropout in RNN and FC layers', default=0.15)
    parser.add_argument('--embedding-size', type=int, help='Embedding size', default=64)
    parser.add_argument('--hidden-size', type=int, help='Hidden state size', default=128)
    parser.add_argument('--n-layers', type=int, help='Number of recurrent layers', default=1)
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional RNN in the encoder')
    parser.add_argument('--clipping', type=float, help='Gradient clipping', default=0.25)
    parser.add_argument('--share-embeddings', action='store_true', help='Share input and output embeddings')
    parser.add_argument('--no-pytorch-rnn', action='store_true', help='Use our hand-written RNN implementations'
                                                                      '(considerably less efficient) instead of the'
                                                                      'PyTorch ones PyTorch RNN implementations'
                                                                      'instead of our hand-written ones, for'
                                                                      'efficiency.')
     ```
     

```
For evaluating a model:
```
python src/cnn/evaluate.py [options]
```
### Experiments highlights

#### Quantitative analysis
The accuracies obtained with this model in each subset are the following ones: 78.97% on train, 57.02% on validation and 56.56% on test.

We observe that:

  

- `numbers__place_value` is extremely easy for our model, because it
    is almost a character-level task. The model is asked to return the
    nth digit of a given number. So, it has to understand to each digit
    it refers the question, but this involves no high-level, abstract
    tasks.

- `numbers__round_number` is the second best-performing task and,
again, it basically consists of a character-level task (rounding to
k digits).

- The other problems are all related to number theory and require a
deeper understanding, so accuracies are lower. At least, all of them
are related, so the model can easily benefit from multi-task
learning and might be modular up to a certain point. The accuracies
of each one of this tasks seem to be correlated to the intrinsic
difficulty of each task, especially because some of them depend on
the skills required for other ones in a composite manner. For
instance, just knowing that a given number is a factor of another
one is the easiest task (70.32%). However, specifically retrieving
the greatest common divisor is slightly more difficult (60.61%).
Perhaps not surprisingly, the latter obtains almost the same
accuracy as the prime detection task (60.25%), an extremely related
task, so we hypothesize that the model might be using almost the
same knowledge for solving these two tasks. Finally, knowing the
exact remainder (numbers\_\_div\_remainder) is the task with the
least performance, because just knowing whether the remainder is
zero or not is considerably easier than retreiving the specific
remainder.

- Unfortunately, we do not have enough evidence to support this claim,
but we believe to be the case that the model will generalize better
to unseen cases (eg. greatest numbers than seen in training) with
the number-theory tasks than the easier, character-level ones.

 #### Qualitative analysis
 
In training, it is interesting to see the evolution of answers to
True/False questions. From very early in the training, the model seems
to be able to detect that some questions require True/False answers, and
it starts by writing either ’t’ or ’f’, but it is not able of accurately
predicting whether they will be true or false. Because of teacher
forcing, "frue" or "talse" are very frequent answers in the first
epochs:

    Source: is 505913 prime?
    Hypothesis: talse
    Target: false
    Accuracy += 0
    
    Source: is 9029 a prime number?
    Hypothesis: frue
    Target: true
    Accuracy += 0

Detecting that a question requires a boolean answer or that
`P(’alse’|’f’) = 1` (in this dataset) is relatively easy, but
understanding the question, or even more, guessing its answer, is not.
Later in the training, once the model has higher accuracies, these
artifacts caused by teacher forcing disappear. Once the model is fully
trained, wrong answers to True/False will almost always be either "true"
or "false", which will be incorrect but at least plausible answers.

In addition, we make the following observations on the selected model in
the test set:

  - `numbers__base_conversion`: In the wrong answers of this problem, we
    observe that at least the model gives plausible answers. For
    instance, when is asked to convert a given number to binary, the
    output is, indeed, binary:
    
        Question: 32 (base 4) to base 2
        Hypothesis: 10110
        Target: 1110<EOS>
    
    Recall that in our implementation we cut the decoding when the
    hypothesis has reached the target length (we do not wait until it
    produces an `<EOS>`, because we can already assure that it is
    wrong).

  - `numbers__div_remainder`: The model does not do well in this task.
    At least, the errors are relatively plausible answers (eg. they are
    numbers relatively small in questions that expect small numbers):
    
        Question: calculate the remainder when 1799 is divided by 1796.
        Hypothesis: 11
        Target: 3<EOS>
    
    We do not find any pattern in the right answers.

  - `numbers__gcd`: In this case, we said that the model performs well.
    A pattern that we have found in the errors is that at least it gets
    the parity right:
    
        Question: calculate the greatest common factor of 56 and 20.
        Hypothesis: 8<EOS>
        Target: 4<EOS>
        
        Question: what is the highest common divisor of 300 and 720?
        Hypothesis: 40<EOS>
        Target: 60<EOS>

  - `numbers__is_factor`: As we said, all the wrong answers are at least
    proper boolean answers. We have found that the model is more prone
    to failing with bigger numbers:
    
        Question: does 13 divide 23767?
        Hypothesis: true<EOS> 
        Target: false<EOS>

  - `numbers__is_prime`: The model might have learned non-trivial
    factorizations. The model could just learn to output that an even
    number is composite (except 2) and an odd number is prime. However,
    in the dataset there are not many even numbers (the authors of the
    dataset did not want to make the task too easy), and recall from the
    quantitative analysis that the accuracy in `numbers__is_prime` in
    the test test is greater than 60.25. Qualitatively, we observe
    non-trivial implicit factorizations such as:
    
        Question: is 20783 composite?
        Hypothesis: true<EOS>
        Target: true<EOS>
    
    The factors of 20783 are 1, 7, 2969, 20783. This particular case
    could be just luck, but there are other similar cases. It could be
    the case that the model does not know all the factors, but at least
    it may be understanding that, say, the number is divisible by 7.

  - `numbers__lcm`: In this case, there are some remarkable answers,
    such as:
    
        Question: find the common denominator of -115/6 and 5/21.
        Hypothesis: 42<EOS>
        Target: 42<EOS>
    
    We do not know if the model is actually solving the task or has
    found some proxy, but at least this heuristic must be actually
    related to number theory, because we observe no spurious
    correlations between the inputs and the outputs.

  - `numbers__list_prime_factors`: The model did not perform well in
    this task. However, there are remarkable (yet, unfortunately, not
    very common) successes, such as:
    
        Question: list the prime factors of 708.
        Hypothesis: 2, 3, 59<EOS>
        Target: 2, 3, 59<EOS>
    
    However, these answers are not anecdotic, because it is difficult to
    guess them just by chance and the accuracy is significantly better
    than 0 (32.46%)..

  - `numbers__place_value`: Recall from the quantitive analysis that for
    this kind of problems our best model obtained almost an accuracy of
    100%. We believe to be interesting to perform an error analysis, to
    perform whether the wrong answers follow a certain pattern. Indeed,
    they do. Most of the wrong answers have one thing in common: the
    question asks to retrieve a digit very close to the right of a big a
    number. This seems to a certain overfitting to the specific digit
    lengths of the dataset, as we said in the quantitative analysis. For
    instance:
    
        Question: what is the units digit of 39358?
        Hypothesis: 9<EOS>
        Target: 8<EOS>
    
    Notice that the selected model is not bidirectional, which might be
    negative for these cases (but, empirically, bidirectional models did
    not outperform the others in the overall dataset).
    
      - `numbers__round_number`: We observe that there are many correct
    answers with relatively long sequences of digits:
    
        Question: round 0.0000083 to six dps.
        Hypothesis: 0.000008<EOS>
        Target: 0.000008<EOS>


## Transfer learning

Best accuracy in Mit67: 83.02, with ResNet-50 pre-trained on Places. TODO: README.
