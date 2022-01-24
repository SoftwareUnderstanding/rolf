## Local scene-recognition model training

#### Train GoogleNet InceptionV3 model

https://arxiv.org/abs/1512.00567

`python local_cnn_train.py --cnn_model=inception --data_directory=places205-subset27 --batch_size=16 --img_size=224 --first_training_epochs=10 --second_training_epochs=20`

#### Train DenseNet121 model

https://arxiv.org/abs/1608.06993

`python local_cnn_train.py --cnn_model=densenet --data_directory=places205-subset27 --batch_size=16 --img_size=224 --first_training_epochs=10 --second_training_epochs=20`


#### Google Cloud Training

`gcloud ml-engine jobs submit training $JOB_ID --job-dir gs://licenta-storage/jobs/$JOB_ID --module-name cloud-trainer.train --package-path ./cloud-trainer --region us-east1 --config=trainer/cloudml_gpu.yaml -- --data_bucket gs://licenta-storage/ --data_file places-dataset.h5`

## Local event-recognition model training

`python oc-cnn_local_train.py --data_directory=./newWIDER --context_model_file=../context-recognition/densenet121-model.h5 --training_epochs=20`

#### Google Cloud Training

`gcloud ml-engine jobs submit training job0_12 --job-dir gs://licenta-storage/jobs-events/job0_12 --module-name cloud-trainer.train --package-path ./cloud-trainer --region us-east1 --config=cloud-trainer/cloudml_gpu.yaml -- --data_bucket gs://licenta-storage/ --data_file wider-dataset.h5 --context_model_file densenet121-model.h5`

## Local event-recognition model serving

`python run_keras_server.py --pretrained_model=../local-activity-recognition-model.h5`

Setup:
$ pip install -U flask-cors

