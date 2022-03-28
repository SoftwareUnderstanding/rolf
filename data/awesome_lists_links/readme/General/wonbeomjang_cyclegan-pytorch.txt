# cyclegan-pytorch
This repository is a deep-running part of a program that changes the style of a photo. And I refer to the following paper.
<p>[A reference article] (https://arxiv.org/abs/1703.10593)</p>

## preparing dataset
We used the supplied cycleGAN dataset.You can download datasets from this link.(https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)  

organize image following structure
example) monet2photo
```
dataset/
    photo/
        0001.jpg
        0002.jpg
        0003.jpg
        0004.jpg
        ...
    monet/
        0001.jpg
        0002.jpg
        0003.jpg
        0004.jpg
        ...
```

or run python file
```bash
python download_dataset.py --style monet2photo
```
## how to train
run main file
example) monet2photo
```bash
python main.py --style monet2photo
```

## how to test
```bash
python test.py --param_path PATH/TO/THE/PARAMETER --input_dir PAHT/TO/THE/INPUT/DIRECTORY --output_dir PATH/TO/THE/OUTPUT/DIRECTORY
```

## Sample
![](images/sample.png)
