# you only look once
YOLO v3 algorithm implemented with PyTorch

# dependencies
- PyTorch=1.0
- numpy
- pandas
- matplotlib
- opencv

# instruction
1. download weight file at this site(https://pjreddie.com/media/files/yolov3.weights), which is based on COCO dataset.
2. In terminal(source activate your pytorch env first): `python detector.py --images xxxx.png(your image file)  --det det` to do object detection on picture.
3. video detection


# references
https://arxiv.org/pdf/1506.02640.pdf \
https://arxiv.org/pdf/1612.08242.pdf \
https://pjreddie.com/media/files/papers/YOLOv3.pdf \
https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/ 
