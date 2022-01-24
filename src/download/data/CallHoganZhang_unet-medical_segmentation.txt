
# 基于unet网络的医学分割项目

	因为对医学分割感兴趣，所以这里就使用Unet了解这相关方向。
	
	之前是使用Keras作眼球血管分割，而这里就沿用Pytorch作另外的项目。

### Unet
	1.这是基于Pytorch框架下的Uet网络，在医学分割上经常使用。这里就实现一个简单的Unet网络
	
	目的在于熟悉Pytorch框架，dataset的使用以及Unet的原理。
	
	2.训练。你可以通过train.py进行训练，因为数据量少，所以训练的时间很短。在训练的过程也使用了Progbar也查看训练的进度
	
	3.预测。训练完毕后，只要指定测试集以及模型的路径，就可以在输出路径找到模型分割后的图片。

### Tornado部署
	
	1.这里还采用了Tornado框架进行部署，实现在线的图片识别功能。
	
	  但是这里我无法使用Tornado在前端页面展示图片，所以只能另外开启一个服务器来进行展示了。

	2.你可以运行app.py，然后在浏览器中输入你指定的Ip，Port进行测试。app.py是在tornado框架下搭建的服务器。

	3.然后你需要在static文件夹下另外新建一个cmd，输入命令：python -m http.server 8000，或者另外一个与Tornado不一样的端口，用于图片的展示
	
	4.那么每运行成功一次，就会另外在static文件夹下保存你识别的图片
	
### 数据集与模型的链接

	网盘：https://pan.baidu.com/s/12Zzo6vVFeXovSO6eUdMVZQ 
	
	keys：6hs2

### 参考

	https://zhuanlan.zhihu.com/p/37496466
	
	https://arxiv.org/abs/1505.04597
	
	