# SPIRAL++

A PyTorch implementation of [Unsupervised Doodling and Painting with Improved SPIRAL
by Mellor, Park, Ganin et al.](https://arxiv.org/abs/1910.01007)

For further details, see https://learning-to-paint.github.io for paper with generation videos.

### Installing

The easiest way to build and install all of PolyBeast's dependencies
and run it is to use Docker:

```shell
$ docker build -t spiralpp .
$ docker run --name spiralpp -it -p 8888:8888 spiralpp /bin/bash
```

or 

```shell
$ docker run -it -p 8888:8888 urw7rs/spiralpp:latest /bin/bash
```

To run PolyBeast directly on Linux, follow this guide.

#### Linux

Create a new Conda environment, and install PolyBeast's requirements:

```shell
$ conda create -n spiralpp python=3.7
$ conda activate spiralpp
$ pip install -r requirements.txt
```

Install spiral-envs

Install required packages:

```shell
$ apt-get install cmake pkg-config protobuf-compiler libjson-c-dev intltool
$ pip install six setuptools numpy scipy gym
```

**WARNING:** Make sure that you have `cmake` **3.14** or later since we rely
on its capability to find `numpy` libraries.

Install cmake by running:
```shell
$ conda install cmake
```

Finally, run the following command to install the spiral-gym package itself:

```shell
$ git submodule update --init --recursive
$ pip install -e spiral-envs/
```

You will also need to obtain the brush files for the `libmypaint` environment
to work properly. These can be found
[here](https://github.com/mypaint/mypaint-brushes). For example, you can
place them in `third_party` folder like this:

```shell
$ wget -c https://github.com/mypaint/mypaint-brushes/archive/v1.3.0.tar.gz -O - | tar -xz -C third_party
```

Finally, the `Fluid Paint` environment depends on the shaders from the original
`javascript` [implementation](https://github.com/dli/paint). You can obtain
them by running the following commands:

```shell
$ git clone https://github.com/dli/paint third_party/paint
$ patch third_party/paint/shaders/setbristles.frag third_party/paint-setbristles.patch
```

PolyBeast requires installing PyTorch
[from source](https://github.com/pytorch/pytorch#from-source).

PolyBeast also requires gRPC, which can be installed by running:

```shell
$ conda install -c anaconda protobuf
$ ./scripts/install_grpc.sh
```

Compile the C++ parts of PolyBeast:

```
$ pip install nest/
$ export LD_LIBRARY_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib:${LD_LIBRARY_PATH}
$ python setup.py build develop
```

### Running PolyBeast

To start both the environment servers and the learner process, run
```shell
$ python -m torchbeast.polybeast \
     --dataset celeba-hq \
     --env_type libmypaint \
     --canvas_width 64 \
     --use_pressure \
     --use_tca \
     --num_actors 64 \
     --total_steps 30000000 \
     --policy_learning_rate 0.0004 \
     --entropy_cost 0.01 \
     --batch_size 64 \
     --episode_length 40 \
     --xpid example
```

Results are logged to `~/logs/torchbeast/latest` and a checkpoint file is
written to `~/logs/torchbeast/latest/model.tar`.

The environment servers can also be started separately:

```shell
$ python -m torchbeast.polybeast_env --num_actors 10
```

Start another terminal and run:

```shell
$ python -m torchbeast.polybeast --no_start_servers
```

### Testing trained models

The provided [jupyter notebook](notebooks/demo.ipynb) will load checkpoints at a specified path to draw a single sample.

If you're using docker run 

```shell
$ jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

make sure you opened a port for the container.

## Repository contents

`libtorchbeast`: C++ library that allows efficient learner-actor
communication via queueing and batching mechanisms. Some functions are
exported to Python using pybind11. For PolyBeast only.

`nest`: C++ library that allows to manipulate complex
nested structures. Some functions are exported to Python using
pybind11.

`third_party`: Collection of third-party dependencies as Git
submodules. Includes [gRPC](https://grpc.io/).

`torchbeast`: Contains `monobeast.py`, and `polybeast.py` and
`polybeast_env.py`. (`monobeast.py` is currently unavailable)

`spiral-envs`: [spiral-envs](https://github.com/urw7rs/spiral-envs/tree/f4deb68b867a5688eb597902b7086f6914c33901) is a libmypaint and fluidpaint based environments. ported to openai
gym from [spiral](https://github.com/deepmind/spiral/tree/master/spiral/environments).

## License

spiralpp is released under the Apache 2.0 license.
