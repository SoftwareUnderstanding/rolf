# infoganJL
Christian Landeros

Year 3, PhD Student

Harvard-MIT Program in Health Sciences and Technology


InfoGAN implementation in Julia with flexibility generator, descriminator, and auxiliary neural network architectures. InfoGAN model structure is implemented as described in [https://arxiv.org/abs/1606.03657] with semi-supervised training capacity for auxiliary networks. Additionally, the Discriminator output and loss are designed according to Wassterstein GAN innovations [https://arxiv.org/abs/1701.07875]"

![overview of InfoGAN model](fig.png)

#### Running as a script
Run trianing with default settings by running `julia infoGAN.jl`. This will load our experimental digital holography dataset from the resources file, and implement default neural network architectures and learning parameters. Parameters can be customized by providing additional arguments in the initial call.

For example, the sizes of all semisupervised (categorical) codes can be set as follows:

`cd src; julia InfoGAN.jl --c_SS 10 5 3 --epochs 1 --c_cont 5`

By default, InfoGAN.jl reads a digital holography dataset located in ./resources/training_data, and creates ground truth labels for semisupervised codes based on file names. The defual neural network architectures defined by this package are designed to evaluate this dataset, and definitions are located in the helpers.jl file. The default Front-End neural network depends on a version of MAT that builds with Julia v1.0.1:

```
julia> ]
(v1.0.1) pkg> add https://github.com/halleysfifthinc/MAT.jl#v0.7-update
julia > using MAT
[ Info: Precompiling MAT [23992714-dd62-5051-b70f-ba57cb901cac]
```
Other package dependencies are: Knet, Printf, ArgParse, Compat,	GZip, Images, JLD2, SparseArrays,	Statistics, Random.

loaddata() and all default neural network architectures can be modified in the helpers.jl to deal with new datasets when using this package as a script.

#### Running in a notebook
If running InfoGAN module in a notebook environment, neural network archtectures can be defined prior to creating the initial InfoModel structure. Once an InfoModel strucuture is defined from hyperparameters in `o` and from neural network weights/functions, we can run the `train(xtrn, ytrn, xtst, ytst, model; mdlfile=MDLFILE, logfile=LOGFILE, printfolder=PRINTFOLDER)` function to start training. [A demo](src/demo.ipynb) is shown in demo.ipynb.
