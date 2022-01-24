# Pedestrian Detection

Pedestrian Detection using transfer learning with coco model

Steps:
1) Label pedestrian dataset from https://www.cis.upenn.edu/~jshi/ped_html/
2) Tran tesorFlow object detection model with labeled data and pretrained SSD Mobilenet as fine tune checkpoint.(train_model.py)
3) Infer model performance on test data (infer_model.py)
