# Plot_CNNs_Python_Latex

This project was created during my master's thesis. It consists of a Python script that generates latex code that can be compiled in a CNN architecture. My idea was to plot the YOLOv2 architecture as a vectorized graphic. The architectural style should be similar to the YOLOv1 architecture, specified in the paper https://arxiv.org/pdf/1506.02640.

For Latex, you only need the tikz package and to run the python script, I used Python 3.5 with the packages:
  - numpy
  - matplotlib
  - cv2

You can run the script in a command line as usually: 
  python Plot_Network.py
This creates a new folder named "Latex_Files" it contains the compilable Latex file.


If you want to change the network, have a look at the list 'architecture' in the Plot_Network.py file line 6:
```python
  architecture = [{"width" : 416, "height" : 416, "filter" : 3, "kernel_size" : 3},
                {"width" : 208, "height" : 208, "filter" : 32, "kernel_size" : 3},
                {"width" : 104, "height" : 104, "filter" : 64, "kernel_size" : 3},
                {"width" : 52, "height" : 52, "filter" : 128, "kernel_size" : 3},
                {"width" : 26, "height" : 26, "filter" : 256, "kernel_size" : 3},
                {"width" : 13, "height" : 13, "filter" : 512, "kernel_size" : 3},
                {"width" : 13, "height" : 13, "filter" : 3072, "kernel_size" : 3},
                {"width" : 13, "height" : 13, "filter" : 1000, "kernel_size" : 3},
                ]
```
Each entry corresponds to a CNN-Layer, the rest should be self explained. The result looks fairly simple, but as a rough baseline, hopefully it helps someone. :)
