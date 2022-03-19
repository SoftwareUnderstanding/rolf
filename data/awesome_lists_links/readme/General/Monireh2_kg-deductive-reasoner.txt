## End-To-End Memory Networks for Deductive Reasoning over Knowledge Graph

This is a modification of implementation of [MemN2N model in Python](https://github.com/vinhkhuc/MemN2N-babi-python) for the [Deductive Reasoning over Knowledge Graph](https://arxiv.org/abs/1811.04132v1)
as inspired by the Section 4 of the paper "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)". It is based on
Facebook's [Matlab code](https://github.com/facebook/MemNN/tree/master/MemN2N-babi-matlab).

[Web-based Demo](Coming soon!)


## Requirements
* Python 2.7
* Numpy, Flask (only for web-based demo) can be installed via pip:
```
$ sudo pip install -r requirements.txt
```
* [Sample Normalized KG Dataset](https://drive.google.com/file/d/1qwyiGlxyxrRBV7FoZrHAFe_girUZrBxY/view?usp=sharing) should be downloaded and decompressed to `data`:
```

$ mkdir data
=======

$ tar xvf sample_data_normalized.tar.xz -C data

```
* [Sample Json Format Dataset](https://drive.google.com/file/d/1Wc50ul4xIrvGAI9HqlFjVS95CgjPXejJ/view?usp=sharing) should be downloaded and decompressed to data in case you want to run the normalization:
```

$ tar xvf sample_json_files.tar.xz -C data
python json_reader_normalizer.py

```
## Usage
* To run on a knowledge graph reasoning task, use `kg_reasoner_runner.py`. For example,
```
python kg_reasoner_runner.py
```
The output will look like:
```
Using data from data/task_name/task_name
Train and test for task task_name ...
1 | train error: 0.876116 | val error: 0.75
|===================================               | 71% 0.5s
```


## Knowledge Graph Reasoning Demo
* In order to run the Web-based demo using the pretrained model `task_name.pklz` in `trained_model/`, run:
```
python -m demo.qa_kg
```

* Alternatively, you can try the console-based demo:
```
python -m demo.qa_kg -console
```

* The pretrained model `task_name.pklz` can be created by running:
```
python -m demo.qa_kg -train
```

* To show all options, run `python -m demo.qa_kg -h`


### Author

* Monireh Ebrahimi



### References
* Monireh Ebrahimi, Md Kamruzzaman Sarker, Federico Bianchi, Ning Xie, Derek Doran, Pascal Hitzler 
  "[Reasoning over RDF Knowledge Bases using Deep Learning](https://arxiv.org/abs/1811.04132)",
  *arXiv:1811.04132[cs]*.

* Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)",
  *arXiv:1503.08895 [cs.NE]*.
