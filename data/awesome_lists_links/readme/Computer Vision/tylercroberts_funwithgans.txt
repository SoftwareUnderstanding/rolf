# `funwithgans` README  ![](https://travis-ci.com/tylercroberts/funwithgans.svg?branch=master)

This repo is a collection of simple GAN projects. Each sub-project will be self-contained and consists of at least:
- a `networks.py` file, containing code relevant to building the networks needed for the example,
- a `requirements.txt` file, listing any additional package requirements in addition to those indicated in the main `requirements.txt` 
- an `__init__.py` file, which can be called in order to start training of a new model.
- an `out` directory which is used to store any image or data outputs from the training script.

There may be some duplication between the modules in each of these sub-projects such as boiler-plate code for parsing command-line arguments, setting up logging, etc.


### Installation Help:
To run these models, you will need to first install all dependencies.
These can be located in the `requirements.txt` files of the model folder you wish to run.

Move into the model folder and run the following command:

`pip install -r requirements.txt`

To utilize `make` alongside this, you will need to install `funwithgans` itself as a package.
You can do so from the root folder, using the following command which will install an editable version
of the package that you can make changes to and immediately see the effects.

`pip install -e .`

Make will allow you to run the examples with a simpler command, while limiting the options you can change.
They will oftentimes also perform additional steps, like cleaning up the output directories, 
or running other necessary stages.

In general, a `make` command will look something like:

`make dcgan-example`

The subsections below will provide the required `make` commands, if there are any to run the examples.

To remove this package when you are done, and to keep your environment clean, use:

`pip uninstall funwithgans`

## Projects:

### Deep Convolutional Generative Adversarial Network (DCGAN):

[DCGAN Arxiv link here](https://arxiv.org/abs/1511.06434)

To run this example, you will need to call `__init__.py` from the `dcgan` folder.
This can either be done directly with `python dcgan/src/__init__.py`, or through `make`. 

##### `make` command:

`make dcgan-example`

Using the make command will read a `config.json` file found in the `dcgan` directory in order to set flags 
and other command line arguments such as the location of necessary files. Please take a look at `parse_arguments` 
found in `dcgan/utils.py` to identify these arguments and create your config file accordingly.

Contents from a sample config file are below:
```json
   {
    "storage-dir": "data",
    "model-dir": "models",
    "image-dir": "dcgan\\out",
    "log-dir": "logs",
    "reproducible": 0,
    "loader-workers": 2,
    "batch-size": 128,
    "image-dim": 64,
    "epochs": 1,
    "lr": 0.0002,
    "beta": 0.999,
    "ngpu": 1
   }
```

Note that the names **MUST** be identical to those in the parser.





