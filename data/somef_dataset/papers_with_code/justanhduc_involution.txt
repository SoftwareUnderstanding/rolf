# Involution

A native CUDA/C++ Pytorch implementation with Python wrapper of [Involution: Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/abs/2103.06255).

## Features

- This implementation is the same as the official version as all the CUDA codes are taken from there.
Minimal C++ wrapper is written.
- This implementation does not require CuPy as a dependency.
- This implementation supports dilation with `same` padding.
- This implementation supports `Half` floating point (experimental).

## Usage

Firstly, set a `CUDA_HOME` variable to point to your CUDA root.

```
export CUDA_HOME=/path/to/your/CUDA/root
```

Then, simply clone this repo and copy the package to your code base. 
Then just import it from the package

```
from involution import Involution
import torch as T

inv = Involution(16, 3, 1, dilation=1).cuda()
input = T.randn(8, 16, 64, 64).cuda()
print(inv(input).shape)
```

In the first import time, it will compile the package so it will take some time.
From the second time, the import time will be normal.

## Testing

```
cd involution
pytest test.py
```

**Note**: The tests for `fp16` are likely to fail the test.

**Note**: Any value of `dilation` larger than `1` will fail the test. 

## License

MIT. See [here](https://github.com/d-li14/involution/blob/main/LICENSE).

## Reference

[Official code](https://github.com/d-li14/involution)
