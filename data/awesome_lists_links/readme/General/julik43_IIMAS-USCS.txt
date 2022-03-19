# IIMAS-USCS

Authors: Andrew Smith, David Kant, Ivette Velez.

Part of the project: Caleb Rascon, Pablo Rubio, Francisco Neri.

This model is programmed to make a verification of two audio signals and indicate if the second signal has the first signal on it.

It uses a modified ResNet 50. The original residual networks can be found in: https://arxiv.org/abs/1512.03385

It uses three separated databases each one with the desired audio for train, validation and test, the audios loaded in this model must be flac o wav, the databases paths must be configured  at the end of the code, or sent as a parameter (e.g. --train_dir /home/train_db).

To run the model use the next command:

If all the parameters are configured correctly in the file:

python resnet_50_v1.py

If you want to configure one or more parameters use:

python resnet_50_v1.py --learning_rate 0.01 --num_epochs 10 --batch_size 10 --train_dir /train --valid_dir /valid --test_dir /dir

note: you can configure just one parameter writing: --name_of_the_parameter value


