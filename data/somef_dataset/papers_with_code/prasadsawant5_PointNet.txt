# PointNet
This repository is an attempt to develop the object classifier part of the [PointNet](https://arxiv.org/abs/1612.00593) research paper. The classifier is implemented using TensorFlow.

The classifier classifies between 40 objects, which are,
1. airplane
1. bathtub 
3. bed
4. bench 
5. bookshelf
6. bottle
7. bowl 
8. car 
9. chair
10. cone 
11. cup
12. curtain
13. desk
14. door
15. dresser
16. flower_pot 
17. glass_box
18. guitar
19. keyboard
20. lamp
21. laptop
22. mantel
23. monitor
24. night_stand
25. person
26. piano
27. plant
28. radio
29. range_hood
30. sink
31. sofa
32. stairs
33. stool
34. table
35. tent
36. toilet
37. tv_stand
38. vase
39. wardrobe 
40. xbox

## Citation
```
@article{qi2016pointnet,
  title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  journal={arXiv preprint arXiv:1612.00593},
  year={2016}
}
```

## Prerequisites
1. Python3
2. TensofFlow
3. Numpy
4. [MobileNet40](https://drive.google.com/uc?id=1l-e6CLERqxLfExwoYlMM0K2Vq4gVrz_r) dataset

## Training
To train a new model, simply run,
```
python run.py
```
Once training is completed, the weights of the trained model are saved in `saved-model/` directory.

## Inference
To perform inference, simply run,
```
python inference.py <file>
```
where `<file>` is the path of the numpy file you would like to perform an inference on.