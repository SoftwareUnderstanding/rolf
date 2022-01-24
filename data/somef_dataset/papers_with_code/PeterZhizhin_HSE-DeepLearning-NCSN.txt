# Noise Conditional Score Networks
## Original paper: https://arxiv.org/abs/1907.05600 

This is a recreation of the original paper using PyTorch.

Original paper code can be found [here](https://github.com/ermongroup/ncsn).

This is a homework for the Deep Learning course at National Research University Higher School of Economics, Moscow Russia.

The work has been done by Petr Zhizhin (piter.zh@gmail.com) and Dayana Savostyanova (dayanamuha@gmail.com).

## Results

The final report about the recreation is available in Russian language.
You can read it in the repository [here](https://github.com/PeterZhizhin/HSE-DeepLearning-NCSN/blob/master/Report/NCSN_homework_report.pdf).

### MNIST

The model on MNIST was trained for 200k iterations on a Tesla V100 for straight ~48 hours.

|![MNIST generated sample](https://raw.githubusercontent.com/PeterZhizhin/HSE-DeepLearning-NCSN/master/samples/mnist_generated_samples.png)|![MNIST generation process](https://raw.githubusercontent.com/PeterZhizhin/HSE-DeepLearning-NCSN/master/samples/mnist_generation_process.png)|
|:-:|:-:|
|MNIST generated samples|MNIST generation process|

### CIFAR

The model on CIFAR was trained on 150k iterations on a Tesla V100 for ~32 hours. The model is underfitted, we were out of budget to train further.

|![CIFAR generated sample](https://raw.githubusercontent.com/PeterZhizhin/HSE-DeepLearning-NCSN/master/samples/cifar_generated_samples.png)|![CIFAR generation process](https://raw.githubusercontent.com/PeterZhizhin/HSE-DeepLearning-NCSN/master/samples/cifar_generation_process.png)|
|:-:|:-:|
|CIFAR generated samples|CIFAR generation process|

## Running training and generation

The model can be trained and evaluated on two datasets: MNIST and CIFAR-10.

### Training

```bash
usage: langevin_images.py [-h] [--dataset DATASET]
                          [--dataset_folder DATASET_FOLDER] [--mode MODE]
                          [--n_generate N_GENERATE] [--download_dataset]
                          [--sigma_start SIGMA_START] [--sigma_end SIGMA_END]
                          [--num_sigmas NUM_SIGMAS] [--batch_size BATCH_SIZE]
                          [--model_path [MODEL_PATH]] [--log [LOG]]
                          [--save_every SAVE_EVERY] [--show_every SHOW_EVERY]
                          [--n_epochs N_EPOCHS]
                          [--show_grid_size SHOW_GRID_SIZE]
                          [--image_dim IMAGE_DIM]
                          [--n_processes [N_PROCESSES]]
                          [--target_device [TARGET_DEVICE]]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET
  --dataset_folder DATASET_FOLDER
  --mode MODE
  --n_generate N_GENERATE
  --download_dataset
  --sigma_start SIGMA_START
  --sigma_end SIGMA_END
  --num_sigmas NUM_SIGMAS
  --batch_size BATCH_SIZE
  --model_path [MODEL_PATH]
  --log [LOG]
  --save_every SAVE_EVERY
  --show_every SHOW_EVERY
  --n_epochs N_EPOCHS
  --show_grid_size SHOW_GRID_SIZE
  --image_dim IMAGE_DIM
  --n_processes [N_PROCESSES]
  --target_device [TARGET_DEVICE]
```

You can look at the default arguments in `langevin_images.py` file.

By default, it trains on the MNIST dataset for 1 epoch on a CPU.

To download MNIST dataset and then train on it using a GPU, while making a checkpoint every 10 epochs and generating sample images every 5 epochs on Tensorboard:
```bash
python3 langevin_images.py --dataset mnist --save_every 10 --show_every 5 --n_epochs 500 --target_device cuda --download_dataset
```

It will create a Tensorboard run under `runs` folder. To monitor training, you can run:
```bash
tensorboard --logdir runs
```

The dataset will be downloaded to a folder named `dataset` (controlled by `--dataset` argument). The checkpoints will be available under `langevin_model` path (controlled by `--model_path` argument).

### Generation

After trainining, you can generate samples using the following command. It will create `.png` images under `langevin_model/generated_images`.
```bash
python3 langevin_images.py --dataset mnist --mode evaluate --n_generate 100
```

If you want to create a 16x16 images grid (as in the example above), you can run this command:
```bash
python3 langevin_images.py --dataset mnist --mode generate_images_grid --show_grid_size 16
```

If you want to create an image with the annealing process (as in the example above), you can run:
```bash
python3 langevin_images.py --dataset mnist --mode generation_process --n_generate 8
```

## Checkpoints

You can get the last checkpoints here:

|Dataset|Link|
|-|-|
|MNIST|[000451.pth](https://drive.google.com/file/d/1opf9zUHF3pR1pjOYQVyPxmfhv0aSoakb/view?usp=sharing)|
|CIFAR|[000231.pth](https://drive.google.com/file/d/1JWx4P0JAN7z0TBGoINUaQ9hbXPv8Op8B/view?usp=sharing)|

After you downloaded the checkpoints, put them in your checkpoints folder (`langevin_model`) by default. You also need to create an empty `.valid` file with the same name as your checkpoints. It tells the program that the checkpoint is written fully and is safe to be used.

## Contributions

The contributions are always welcome. We would highly appreciate the following contributions:
1. If you have resources to train the CIFAR model, please provide a better checkpoint. The model __should__ converge to a better result.
2. If you want to do a code cleanup, this would be highly appreciated.
