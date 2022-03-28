# CycleGAN-android
Style transfer on Android with CycleGAN in TensorFlow, applying remote inference on a php server using [vanhuyz's](https://github.com/vanhuyz/CycleGAN-TensorFlow) implementation.

# Pretrained models
My pretrained models: [Monet](https://mega.nz/#F!vZ51yQBR!aDzWzf9jDgoegUsFXpd7LQ), [Cezanne](https://mega.nz/#F!iN5xASAK!i7vRSn_QEkC8ahxnzv5F9w), [Ukyo-e](https://mega.nz/#F!mNhj2IJJ!_xU6BoD4f8B8XstsW6CDSw), [Vangogh](https://mega.nz/#F!iJxVCCAD!g8FZFmOjBHdFv8zvkZH7YA).

Checkpoints: [Monet](https://mega.nz/#F!bBRk2AoA!_7Shwc6MNAIodnDPmz5BAQ), [Cezanne](https://mega.nz/#F!CBpXEAqI!LHzoynceBJxCL8Pqc92CeA), [Ukyo-e](https://mega.nz/#F!iVR0ACRQ!gB2F9wYrgZv6RitQ7h-Hzw), [Vangogh](https://mega.nz/#F!LMgTXCpS!PtZ87RTay-SMVwRo_TO_9Q).

# Environment
- TensorFlow 1.0.0
- Python 3.6.0
- Android Studio
- MAMP

Currently it is not possible to import the .pb model directly into Android, nor convert it to .tflite because Tensorflow Mobile doesn't support the map_fn node yet.

Unsupported TensorFlow Lite Ops:
1. NonMaxSuppressionV2
2. TensorArrayGatherV3
3. TensorArrayReadV3
4. TensorArrayScatterV3
5. TensorArraySizeV3
6. TensorArrayV3
7. TensorArrayWriteV3

# Result

![alt text](https://github.com/matteodalessio/CycleGAN-android/blob/master/pic/collage.jpg)

# Reference

- CycleGAN implementation: https://github.com/vanhuyz/CycleGAN-TensorFlow
- CycleGAN paper: https://arxiv.org/abs/1703.10593
