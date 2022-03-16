# Demo

The trained models are found in resuls/models/demo/ folder. You can retrain the models or use you own if you wish.

There are two ways for using the demo script. 

You can provide a raw.githubusercontent link to the readme that you want to classify by defining the --readme_url paramereter. 
The other way is to define the path of a text file by --text_file which contains the text that you want to classify.

A threshold can be also given for the probability of predicting positiva samples with --threshold. The default value is 0.5.

For example:

```
python3 src/demo/demo.py --models_dir results/models/demo/ --readme_url https://raw.githubusercontent.com/dgarijo/Widoco/master/README.md --threshold 0.7
```