# capgen
> A Neural Image Caption Generator


## Results

This model was trained on the MSCOCO train2014 dataset and obtains the following results

| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr|
| ---| --- | --- | --- | --- | --- | ---| 
| 0.651 | 0.467 | 0.322 | 0.218 | 0.212 | 0.479 | 0.690|

## Getting Started

  1. Clone the repo

    $ git clone https://github.com/tacotaha/capgen.git && cd capgen

  2. Download MS-COCO Training Data

    $ cd scripts && ./get_data.sh

  3. Resize Training Images

    $ ./resize.sh

## Training the model 

Use `src/train.py` to train the model

`$ python src/train.py`


## Evaluate the Model

Use the `eval.sh` script to evaluate the model. The resulting captions can be found in `data/results.json`.

`$ scripts/eval.sh`

## Inference

Use `inference.py` to test the model with an example image

`$ python src/inference.py <img_path.png>`

## A Sample of Results

<img src="img/collage.png">


## References
* <a href="https://arxiv.org/pdf/1411.4555.pdf">Show and Tell: A Neural Image Caption Generator</a>
* <a href="https://arxiv.org/abs/1512.03385" >Deep Residual Learning for Image Recognition</a>
* <a href="http://cocodataset.org/#home"> MSCOCO Dataset</a>
* <a href="https://github.com/lyatdawn/Show-and-Tell">TensorFlow Implementation</a>
* <a href="https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning"> PyTorch Implementation</a>
