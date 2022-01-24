# Shark-PrioritizedExperienceReplay

[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

Inspired by https://github.com/takoika/PrioritizedExperienceReplay,
we provide a fast (cpp-version) implementation of prioritized experience replay buffer.

for a full usage, please take a forward step to https://github.com/7starsea/shark.

## Note
Implementation here is simplified version of PrioritizedExperienceReplay in project https://github.com/7starsea/shark by removing unnecessary bundle.

## Requirement
* python3, (tested on 3.6, 3.7)
* pybind11, (tested on 2.4.3)
* cmake3, (tested on 3.14)
* scikit-build, (tested on 0.10.0), optional

(we use ananconda enviroment for testing)

## Compile [tested on ubuntu]
on linux, you can type `./complie.sh` or `python setup.py build --inplace`.

## Reference
[1] "Prioritized Experienced Replay", http://arxiv.org/abs/1511.05952 

