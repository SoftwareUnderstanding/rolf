# Barlow Twins

Unofficial Tensorflow 2 implementation of the [Barlow Twins Self-Supervised Learning method](https://arxiv.org/abs/2103.03230)

```bash
$ python train.py --name my_test /data
$ python train.py --help
```

```python
model = barlow_twins.BarlowTwinsModel(input_height=224,
                                      input_width=224,
                                      projection_units=8192,
                                      drop_projection_layer=True)
model.load_weights(saved_weights, by_name=True)
# Input image values should be in range [0, 255] --> preprocessing is built into the model
embedding = model(image)
```

# Results

**Convergence (Oxford 102 Flowers**

<img src="art/oxford_flowers_adam_sgd.png" width="600" alt="training_losses"></a>

# Setup

## Pip/Conda

```bash
pip install -r requirements.txt
```

## Docker

**Build**

```bash
docker build -t barlow .
```

**Run a training**

```bash
docker run --rm \
           -t \
           -u $(id -u):$(id -g) \
           --gpus all \
           -v $(pwd):/code \
           -v <DATASET_FOLDER_PATH>:/data \
           -w /code \
           barlow \
           python train.py --name my_test /data
```

# Citations

```bibtex
@article{DBLP:journals/corr/abs-2103-03230,
    author    = {Jure Zbontar and Li Jing and Ishan Misra and Yann LeCun and St{\'{e}}phane Deny},
    title     = {Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
    journal   = {CoRR},
    volume    = {abs/2103.03230},
    year      = {2021},
    url       = {https://arxiv.org/abs/2103.03230},
    archivePrefix = {arXiv},
    eprint    = {2103.03230},
    timestamp = {Mon, 15 Mar 2021 17:30:55 +0100},
    biburl    = {https://dblp.org/rec/journals/corr/abs-2103-03230.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

# TODOs

- Evaluation
    - Linear evaluation
    - KNN eval
- Choose or use custom backbone
- Save model
    - Save only when loss improved
