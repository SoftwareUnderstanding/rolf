# NeMO-NET
Neural Multi-Modal Observation & Training Network for Global Coral Reef Assessment
Created by: Alan Li
Email: alan.s.li@nasa.gov

Startup notes:
1) Installation notes:
  - Install CUDA and GPU Acceleration (version 8 preferable) for NVIDIA cards, if applicable (CUDA 9 has been verified to work with updated Tensorflow v 1.5, with a few deprecation warnings)
  - Install OpenCV. Instructions for full mac install here: http://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5/
  - Make sure the following Python packages are installed: numpy, matplotlib, jupyter, tensorflow, PIL, sklearn, pandas
  - Install osgeo/gdal for geospatial manipulation (required for .shp, .gml, .gtif files): http://www.kyngchaos.com/software/frameworks
  - Install Keras version 2.0.8
  - Install hyperopt and hyperas (pip install)
  - Install pydensecrf, if fully connected CRFs are used (requires cython and rc.exe + rcdll.dll from C:\Program Files (x86)\Windows Kits\8.0\bin\x86 to C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin). This might be a complicated process, as it requires to make the original C++ code with the wrapper in python.
  - Downgrade networkx to 1.11 (pip install networkx==1.11)
  - Additional resources: https://www.lfd.uci.edu/~gohlke/pythonlibs for .whl files

2) Images can be downloaded per request, and are to be contained within the ./Images/ folder. If choosing a different folder, please specify locations within the .yml files

3) Additional resources:
Code
  - VGG16 FCN code and base level code based upon: https://github.com/JihongJu/keras-fcn
  - ResNet architecture loosely based upon: https://github.com/raghakot/keras-resnet
  - Hyperas/ Hyperopt code: https://github.com/maxpumperla/hyperas
  - DeepLab code based upon: https://github.com/DrSleep/tensorflow-deeplab-lfov
  - pydensecrf code found here: https://github.com/lucasb-eyer/pydensecrf

 Papers
  - DeepLab: https://arxiv.org/abs/1606.00915
  - VGG FCN-like: http://www.mdpi.com/2072-4292/9/5/498
  - DCNNs with CRFs: https://arxiv.org/abs/1412.7062
  - VGG: https://arxiv.org/abs/1409.1556
  - ResNet: https://arxiv.org/abs/1512.03385
