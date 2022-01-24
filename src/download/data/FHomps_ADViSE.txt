This repository contains the programming work for my Master's project of 2020 at Keio University, in Dr. Ishigami's Space Robotics Group.

# ADViSE - Assisted Descent using Visual Slope Estimation

ADViSE is a collection of tools intended to process satellite views of Martian terrain.
Its final objective is to provide an algorithm for descent planning for potential Martian landers which, based solely on visual data, can determine bad spots to avoid during descent.

To that end, we intend to create a neural network that can successfully identify such zones, with concern for performance and resolution-agnosticism as the lander gets closer to the ground.
Training is done using data augmentation on already existing DTMs from Hirise and (yet to come) completely synthetic training data rendered with Unity.

## Credits

Keio University Space Robotics Group: http://www.srg.mech.keio.ac.jp/

Pix2Pix GAN for image detection:
* Authors: Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros: https://arxiv.org/abs/1611.07004
* Implementation adapted from Erik Linder-Norén: https://github.com/eriklindernoren/PyTorch-GAN

Martian DTMs (Digital Terrain Models) from University of Arizona's HiRISE: https://www.uahirise.org/dtm/
