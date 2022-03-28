# AE-StyleGAN: Improved Training of Style-Based Auto-Encoders

[[arXiv]](https://arxiv.org/pdf/2110.08718)
[[pdf]](https://www.dropbox.com/s/qxrkcu1d4618w0w/AE-StyleGAN.pdf?dl=0)
[[supp]](https://www.dropbox.com/s/6a4odr09mxgf6z8/AE-StyleGAN_supp.pdf?dl=0)

![Model Architecture.](ae-stylegan_1.png)
![Reconstruction.](ae-stylegan_2.png)

The code is heavily based on [Rosinality's PyTorch implementation of StyleGAN2](https://github.com/rosinality/stylegan2-pytorch).

# Training
All experiments are conducted at 128x128 resolution.

## Decoupled AE-StyleGAN
To train the Decoupled AE-StyleGAN, we use the following training procedure:
```
python train_aegan.py \
--path data/ffhq \
--sample_cache data/sample_ffhq128_64.npy \
--which_latent w_plus \
--lambda_rec_w 0 \
--iter 200000 \
--size 128 \
--name ffhq_aegan_wplus_decoupled \
--log_every 500 \
--save_every 2000 \
--eval_every 2000 \
--dataset imagefolder \
--inception inception_ffhq128.pkl \
--n_sample_fid 10000 \
--decouple_d \
--lambda_rec_d 0 \
--g_reg_every 0 \
--batch 16 \
--lr 0.0025 \
--r1 0.2048 \
--ema_kimg 5 \
--which_metric fid_sample fid_recon --use_adaptive_weight --disc_iter_start 30000
```

## Joint AE-StyleGAN
To train the Joint AE-StyleGAN, we use the following training procedure:
```
python train_aegan.py \
--path data/ffhq \
--sample_cache data/sample_ffhq128_64.npy \
--iter 200000 \
--size 128 \
--name ffhq_aegan_wplus_joint \
--which_latent w_plus \
--lambda_rec_w 0 \
--log_every 500 \
--save_every 2000 \
--eval_every 2000 \
--dataset imagefolder \
--inception inception_ffhq128.pkl \
--n_sample_fid 10000 \
--lambda_rec_d 0.1 \
--lambda_fake_d 0.9 \
--lambda_fake_g 0.9 \
--joint \  # joint train G with D
--g_reg_every 0 \
--batch 16 \
--lr 0.0025 \
--r1 0.2048 \
--ema_kimg 5 \
--which_metric fid_sample fid_recon --use_adaptive_weight --disc_iter_start 30000
```

## Baselines
To train a StyleGAN2 (without R1 regularization), we use the following training procedure:
```
python train.py \
--path data/ffhq \
--iter 200000 \
--size 128 \
--name ffhq_gan \
--log_every 500 \
--save_every 2000 \
--eval_every 2000 \
--dataset imagefolder \
--inception inception_ffhq128.pkl \
--n_sample_fid 10000 \
--g_reg_every 0 \
--batch 16 \
--lr 0.0025 \
--r1 0.2048 \
--ema_kimg 5 
```

To train a (reimplemented) Style-ALAE model, use the following command:
```
python train_alae.py \
--path data/ffhq \
--sample_cache data/sample_ffhq128_64.npy \
--iter 200000 \
--size 128 \
--name ffhq_alae_wtied_recw=1_mlpd=4 \
--which_latent w_tied \
--which_phi_e lin1 \
--n_mlp_d 4 \
--log_every 500 \
--save_every 2000 \
--eval_every 2000 \
--dataset imagefolder \
--inception inception_ffhq128.pkl \
--n_sample_fid 10000 \
--lambda_rec_w 1 \
--lambda_fake_d 1 \
--lambda_fake_g 1 \
--lambda_rec_d 0 \
--lambda_pix 0 \
--lambda_vgg 0 \
--lambda_adv 0 \
--g_reg_every 0 \
--batch 16 \
--lr 0.0025 \
--r1 0.2048 \
--ema_kimg 5 \
--which_metric fid_sample fid_recon
```

# Pretrained Models

Coming soon.

# Citation
If you use this code, please cite
```
@article{han2021ae,
  title={AE-StyleGAN: Improved Training of Style-Based Auto-Encoders},
  author={Han, Ligong and Musunuri, Sri Harsha and Min, Martin Renqiang and Gao, Ruijiang and Tian, Yu and Metaxas, Dimitris},
  journal={arXiv preprint arXiv:2110.08718},
  year={2021}
}
```
