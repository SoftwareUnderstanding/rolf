# Siren - Sinusoidal Representation Networks

Unofficial implementation of 'Implicit Neural Representations with Periodic Activation Functions' using **mxnet gluon** for the model.


Video: https://www.youtube.com/watch?v=Q2fLWGBeaiI

Original project page: https://vsitzmann.github.io/siren/

Paper: https://arxiv.org/abs/2006.09661

Official implementation: https://github.com/vsitzmann/siren

## Reproducing Results

### Installation

Install all packages with pip, run 
```bash
pip install -r requirements.txt
```

### Training

To train the model run the following command 

```
python siren_trainer.py --image_path 'Image Path'
python siren_trainer.py --image_path dog.png
```


### Testing

To test the model run the following command 

```
python siren_tester.py --model_path 'Model Path'
python siren_tester.py --model_path "./checkpoints/model_1.params"
```

'siren_tester.py' will create new image in the root folder 'result.png'
