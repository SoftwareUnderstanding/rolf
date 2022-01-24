# PixelCNN
A simple pytorch implementation of PixelCNN as described in https://arxiv.org/pdf/1601.06759.pdf.

The code is specific to binary data, and in particular to the BinarizedMNIST dataset. 
Still, it can be modified easily to handle multinomial data.

To train the model and produce images out of it, run the following
```
python main.py
```

Additionally, one may pass some arguments through the command line to modify the hyperparameters of the model.
For instance, run
```
python main.py -hl 5
```
to set the number of `hidden layers` to `5`. 

For the full list of command line arguments, see `args.py`.
