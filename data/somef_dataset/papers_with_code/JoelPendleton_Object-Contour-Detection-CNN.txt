# Edge Detection CNN
Edge Detection CNN using Tensorflow 2 using the U-Net and UNet++ architectures.

* UNet Paper: https://arxiv.org/abs/1505.04597
* UNet++ Paper: https://arxiv.org/abs/1807.10165  

To install the required environment install anaconda and run the following commands:
```conda env create -f environment.yml``` and then ```conda activate tensorflow```

To run the neural network execute the following command in the command line:
```python main.py --[Command] --[Architecture]```

The allowed values for **[Command]** and **[Architecture]** arguments/flags are shown in the table below:

<table>
    <tr>
      <th>Command</th>
      <th>Architecture</th>
    </tr>
    <tr>
      <td>Help</td>
      <td>UNet</td>
    </tr>
    <tr>
      <td>Train</td>
      <td>UNet++</td>
    </tr>
     <tr>
      <td>Summary</td>
    </tr>
     <tr>
      <td>Evaluate</td>    
      </tr>
     <tr>
      <td>Predict</td>
    </tr>
</table>

**Command** refers to the functionality of the **Architecture** you which the utilise.

* **Train** trains the model on the training set.
* **Summary** outputs a summary of the architecture of the model.
* **Evaluate** gives the accuracy of the model on the test set.
* **Predict** generates images of the edges as predicted by the model.


To extract the parameters from the prediction produced by the network, run the following command from the Utilities/ folder:
```python draw_lines.py```

The lines are found using the Hough Transform the variable called **threshold** in draw_lines.py 
can be adjusted to allow for lines to be found that are closer together. 

The parameters will be outputted into a file called diamonds-data.txt (in terms of volts).
The data found in the line-data.txt file doesn't account for units (it's left in terms of pixel coordinates).

Example of CNN prediction and draw_lines functionality:

<table>
    <tr>
      <th>Prediction</th>
      <th>Drawn Lines</th>
    </tr>
    <tr>
      <td><img src="draw_lines_example_1.png", width = "500px"></td>
      <td><img src="draw_lines_example_2.jpg", width = "500px"></td>
    </tr>
</table>
