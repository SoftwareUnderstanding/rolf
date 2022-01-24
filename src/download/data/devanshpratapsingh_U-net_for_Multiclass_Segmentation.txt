# U-net_for_Multiclass_Segmentation

UNET PAPER: <link>https://arxiv.org/abs/1505.04597</link>

<b>U-Net Architecture</b>
The architecture follows ‘U’ shaped path. It is a unit network architecture which is built upon the fully connected network in CNN.
<br>
<img src="readme_images/unet.png" height="300px" width="350px" align="middle"></img>
<br>
I have used the Oxford IIIT Pet dataset, which consists of three classes:
1. Main object (Cat or Dog)
2. Border
3. Background
<br>
<b>I took only 10 epochs in training, each epoch took almost 1 hr to train, but I've trained the model for just 2 hrs and still the results are pretty good :)</b>
<br><b>The first image is the input, second one is the masked image and third one is the output we got, train the model for longer and you'll get better results.</b>
<img src="readme_images/Abyssinian_31.jpg" height="180px" width="360px" align="middle"></img>
<img src="readme_images/american_bulldog_72.jpg" height="180px" width="360px" align="middle"></img>
<img src="readme_images/american_pit_bull_terrier_61.jpg" height="180px" width="360px" align="middle"></img>
<br>
<h2><b>Dataset:</b><a href="https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbmNEbDd3RTZKQnRuM25ibXlfT1k0ZzRuM1JDUXxBQ3Jtc0tsRHZCN0RCbXJMVWFBLTBuVEdQNUNmNV9aTkZVN0NzQ1N2ZS0wOVdKN1BReGxxbnRIcGptWENTQXV3VFVCbDRUOHFfRXk4bjlBNVpjVHNNOVNZLWJhcVlHTGVZcTZBUXBaVWxzNzhUVEl5UXE4eE9vNA&q=https%3A%2F%2Fwww.robots.ox.ac.uk%2F%7Evgg%2Fdata%2Fpets"> Oxford IIIT Pet dataset</a></h2>
