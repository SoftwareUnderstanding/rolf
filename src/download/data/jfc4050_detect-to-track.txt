# Detect to Track and Track to Detect

This project is loosely based on [this paper](https://arxiv.org/abs/1710.03958). Here is a (probably non-exhaustive) list of differences between this implementation and the paper:
* The paper suggests sampling adjacent frames (so stride 1) to be passed through the model. This implementation instead determines the stride for each example pair by sampling from a discrete laplacian distribution.
* The paper suggests sampling at most 2k images per class, and 10 frames per video to address dominant classes and very long videos. This implementation instead starts by uniformly sampling a class/video, then uniformly sampling from images containing that class/frames from that video. This solves the same problem while maximizing sample diversity within classes/videos.
* Training follows the approximate joint training scheme, instead of using an alternating training scheme. See [this paper](https://arxiv.org/abs/1506.01497) for a summary of training schemes commonly used for two-stage networks. I am very interested in seeing if the use of [a differentiable RoI-Warping layer](https://arxiv.org/pdf/1512.04412.pdf) will significantly improve the performance of this training scheme, so that may be implemented at some point.
* Any gradients related to losses contributed by anchors crossing the image boundary are not used. They are masked out before the backward pass.
* [Focal loss](https://arxiv.org/abs/1708.02002) is used instead of Binary Cross-Entropy loss with Online Hard Example Mining.
* This implementation uses a slightly modified version of the [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) for tubelet linking. The difference between this implementation of the viterbi algorithm vs the version used in the paper is that at each time step an additional "path" is added, beginning at that time step and only containing a single state. This accomodates tubelets beginning in the middle of the sequence. See `detect_to_track/viterbi.py` for additional details.

This project depends on my assorted collection of machine learning utilities, which can be found [here](https://github.com/jfc4050/ml_utils). The library is very immature so please pay close attention to the version number specified in this project's `requirements.txt`.

The following operations are currently only implemented in CUDA (and not CPU) so this project will require an Nvidia GPU to run.
* ROIPool
* PSROIPool
* PointwiseCorrelation
