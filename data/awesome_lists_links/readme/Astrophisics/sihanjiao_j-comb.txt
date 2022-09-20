<div align="center">

# J-comb algorithm

_✨ Baseed on [uvcombine](https://github.com/keflavich/uvcombine) ✨_

</div>

A novel method to combine high-resolution data with large-scale missing information with low-resolution data containing the short spacing.

## Description

### Input:

fits files of low- and high-resolution image,  angular resolution of the input images, pixel size of the input images,

### Output:

fits file of the combined image

### Usage:

Inside python

```
from combine_fuction import combine_2d
combined_2d(lowresolution_map, highresolution_map, combined_map, beam_size, pixel_size，kenel,....)
```

An example is avaliable [here](https://github.com/SihanJiao/J-comb/blob/main/example_omc3.ipynb).


