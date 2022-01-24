# Product Detection with SSD
- Most of the codes are taken from [tensorflow models](https://github.com/tensorflow/models)
- Parameters and architecture followed from [Paper](https://arxiv.org/abs/1512.02325)
- Product data set taken from [Dataset](https://github.com/gulvarol/grocerydataset)

![Teaser Image](https://github.com/chandra411/Product-Detection/blob/master/out.JPG)

## Installations & setting up environment (Please use python3)
	Environment used is: Ubuntu-18.04 LTS, GTX TitanXP 12 GB, 
	- (Optional) create python3 virtual environment and activate it
	- sudo apt-get install protobuf-compiler (Install protobuf-compiler, if you are getting issues pelase do it from source)
	- pip -r install requirments.txt
	- cd src/models/research/
	- protoc object_detection/protos/*.proto --python_out=.
	- export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim (If you want to add to the bash, please add this command to ~/.bashrc)
	- source ~/.bashrc
	- Change directory to the product_detection_chandrasekahr_pati
Download initial weight model from [link](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz) then extract and place it in models folder
## Training 
	- Data preparation
		Use below sh files to create tf records
		sh create_tf_records.sh
		- Generate train.record and test.record and place them in data folder
		- Please provide respective data paths in the create_tf_records.sh file 
	- Training network
		All of the training cofigurations are defined in models/single_anchor_ssd.config, please go through the config file and edit {ropository root path} to your system current location
		- use below sh file to start training
		sh train.sh
		- User tensorbord with --model_dir to monitor training,
			tesorboard --logdir=./checkpoint and navigate to http://ipaddress:6006/

## Testing and Evalution
	- Convert model from checkpoint to frozen inference graph pb file
		sh ckpt_2_pb.sh
		- Please provide best checkpoint path in sh file
	- Testing and Evalution 
		sh evaluation.sh 
		- evaluation will save image2products.json and metrics.json in given output_dir
		- If you want to save bounding box drwan on top of input image, please use --save_im=1 in arguments, images will be saved in --output_dir/out_images


