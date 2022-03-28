## 3 Speaker Separation Dataset Re-Construction Training and Testing For Conv-TasNet 

This is a PyTorch implementation of the [TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation](https://arxiv.org/abs/1809.07454) that was created by (https://github.com/funcwj/conv-tasnet) with some modifications done by Adam Whitaker-Wilson. The dataset composition function was implemented by WASS (https://github.com/mitmedialab/WetlandAvianSourceSeparation) with some modifications done by Adam Whitaker-Wilson.

See (3 Speaker Separation Dataset Re-Construction Training and Testing For Conv-TasNet.pdf) for project details.

### Datasets, Inferences and Models: as described in (3 Speaker Separation Dataset Re-Construction Training and Testing For Conv-TasNet.pdf)

Link to Datasets: 
```
https://drive.google.com/open?id=1_O-LdrIv6sU15sCN-PU82DBY0eIk_iTp
```

Link to Inferences:
```
https://drive.google.com/open?id=1HofvShgimzUkf1gAE_xW5HCpawT92vV9
```

Link to Models:
```
https://drive.google.com/open?id=1oJg8CDvNUhGQ-1Mvdy_aqdV7Ay-H_5B6
```

### Requirements

* see [requirements.txt](requirements.txt) or run:
```
pip3 install -r requirements.txt
```

### Usage

* digital dataset creation: configure [default.yaml](config/composer)
note: a "dummy" model is trained with 1 epoch as a test.

* training: configure [conf.py](nnet/conf.py) and run [train.sh](train.sh)

* inference
```bash
./nnet/separate.py /path/to/checkpoint --input /path/to/mix.scp --gpu 0 > separate.log 2>&1 &
```

* evaluate, calculates cosine-similarity, dynamic time warp and si-snr
```bash
./nnet/compute_si_snr.py /path/to/ref_spk1.scp,/path/to/ref_spk2.scp,/path/to/ref_spk3.scp /path/to/inf_spk1.scp,/path/to/inf_spk2.scp,/path/to/inf_spk3.scp
```

### Reference

Luo Y, Mesgarani N. TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation[J]. arXiv preprint arXiv:1809.07454, 2018.