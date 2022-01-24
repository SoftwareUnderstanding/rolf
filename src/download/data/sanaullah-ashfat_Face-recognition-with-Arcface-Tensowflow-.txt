
# Face-recognition-with-Arcface-Tensowflow-
ArcFace (Additive Angular Margin Loss for Deep Face Recognition, published in CVPR 2019) implemented in Tensorflow 2.0+. This is an unofficial implementation.

You can find original paper here [link] https://arxiv.org/abs/1801.07698



**Working Procedure:**


![recognition](https://user-images.githubusercontent.com/41291499/128629156-05454773-6e33-4678-b54c-5b64f0c08851.png)

Run integrate.py file. Before running, give your data path. After running that file you will be asked to choose the task that you want to do. If you want to Recognition then give input “ recognition” or if you want to save the embedding then give input “register”. Then it will automatically create embedding and save in a pickle file. 
If you want to see the embedding from the pickle file then go to the test file then uncomment 10 to 12 lines then run.When you go for a recognition task if there is any unknown person and you want to add on your embedding file for later recognition, you will be asked when you run the integrate.py file. Press Y or y to confirm that you want and then give an ID or name then it will be added in your embedding file. 



![face](https://user-images.githubusercontent.com/41291499/128629138-39376241-88fc-4d2c-b42a-86c41287942e.png)

**Models:**

Download the Pre-train model from this link:  https://drive.google.com/file/d/1HasWQb86s4xSYy36YbmhRELg9LBmvhvt/view?usp=sharing
Now paste it to the checkpoints folder.

Test in postman:
If you want to test using postman just run app.py then send bas64 data to the url. 

You can find the official implementation here:   https://github.com/deepinsight/insightface
