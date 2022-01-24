# Detection-Using-Convolutional-Networks
 OverFeat: Integrated Recognition, Localization and Detection Using Convolutional Networks. arXiv:1312.6229 [Cs], December, 2013.


#### Apple 의 An On-device Deep Neural Network for Face Detection 에 사용 및 참고된 논문을 구현하는 프로젝트입니다.
- Vol. 1, Issue 7 ∙ November 2017 
- By Computer Vision Machine Learning Team
- Paper Link : https://arxiv.org/pdf/1312.6229.pdf
- Journal Link: https://machinelearning.apple.com/2017/11/16/face-detection.html

#### Issues List 

- Issue => Go to [https://github.com/LeeGitaek/Detection-Using-Convolutional-Networks/issues/]

## Model Structure

- Layer 1
   => Conv + Max Pooling, <br>
      Channel => 96, <br>
      Filter size (Kernel) => 11 * 11, <br>
      Conv Stride => 4 * 4, <br>
      Pooling Size => 2 * 2, <br>
      Pooling Stride => 2 * 2, <br>
      Zero Padding Size => -, <br>
      Spartial Input Size => 231 * 231 <br>
      
      
- Layer 2
   => Conv + Max Pooling, <br>
      Channel => 256, <br>
      Filter size (Kernel) => 5 * 5, <br>
      Conv Stride => 1 * 1, <br>
      Pooling Size => 2 * 2, <br>
      Pooling Stride => 2 * 2, <br>
      Zero Padding Size => -, <br>
      Spartial Input Size => 24 * 24 <br>
      
      
- Layer 3
   => Conv, <br>
      Channel => 512, <br>
      Filter size (Kernel) => 3 * 3, <br>
      Conv Stride => 1 * 1, <br>
      Pooling Size => -, <br>
      Pooling Stride => -, <br>
      Zero Padding Size => 1 * 1 * 1 * 1, <br>
      Spartial Input Size => 12 * 12 <br>
      
- Layer 4
   => Conv, <br>
      Channel => 1024, <br>
      Filter size (Kernel) => 3 * 3, <br>
      Conv Stride => 1 * 1, <br>
      Pooling Size => -, <br>
      Pooling Stride => -, <br>
      Zero Padding Size => 1 * 1 * 1 * 1, <br>
      Spartial Input Size => 12 * 12 <br>
      
- Layer 5
   => Conv + Max Pooling, <br>
      Channel => 1024, <br>
      Filter size (Kernel) => 3 * 3, <br>
      Conv Stride => 1 * 1, <br>
      Pooling Size => 2 * 2, <br>
      Pooling Stride => 2 * 2, <br>
      Zero Padding Size => 1 * 1 * 1 * 1, <br>
      Spartial Input Size => 12 * 12 <br>
     
#### FC ( Fully Connected Layer )
- Layer 6
   => Full , <br>
      Channel => 3072, <br>
      Filter size (Kernel) => -, <br>
      Conv Stride => -, <br>
      Pooling Size => -, <br>
      Pooling Stride => -, <br>
      Zero Padding Size => -, <br>
      Spartial Input Size => 6 * 6 <br>
   
- Layer 7
   => Full , <br>
      Channel => 4096, <br>
      Filter size (Kernel) => -, <br>
      Conv Stride => -, <br>
      Pooling Size => -, <br>
      Pooling Stride => -, <br>
      Zero Padding Size => -, <br>
      Spartial Input Size => 1 * 1 <br>
      
#### Output Layer 
- Layer 8
   => Full ,
      Channel => 1000, <br>
      Filter size (Kernel) => -, <br>
      Conv Stride => -, <br>
      Pooling Size => -, <br>
      Pooling Stride => -, <br>
      Zero Padding Size => -, <br>
      Spartial Input Size => 1 * 1 <br>
