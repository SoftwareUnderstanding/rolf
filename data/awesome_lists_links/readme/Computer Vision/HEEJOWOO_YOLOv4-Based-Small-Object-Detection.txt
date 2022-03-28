# YOLOv4-Based-Small-Object-Detection   

  * YOLO v4 : https://arxiv.org/abs/2004.10934
  * STAR-Net : http://www.bmva.org/bmvc/2016/papers/paper043/paper043.pdf
  * 건설 설계 도면 내 자체 데이터 셋을 구축한 뒤 YOLOv4와 OCR RARE를 통해 객체 검출 및 문자 인식 파이프라인을 설계하여 철강재료를 적산할 수 있는 기술 개발
  * 플랫폼 및 사용 기술 : Anaconda, Pytorch, OpenCV, Numpy, openpyxl
  * 자체 Custom 데이터 셋 구축
    * 객체 검출 셋
        * train : 761장, val : 26장, test : 80장
    * 문자 인식 셋
        * train : 1만장, val : 3천장, test : 4천장
  * YOLOv4, OCR RARE Network 분석
  * 데이터 복제 증강 기법 및 공간 집중 모듈을 활용한 큰 도면 내 소형 객체 검출 성능 향상
  * 객체 검출 및 문자 인식 파이프라인 설계

    
   
## 철강자재 검출 및 문자인식 전체 과정
![전체 과정 설명](https://user-images.githubusercontent.com/61686244/143766402-556fdbdb-91c0-4daf-96f4-fe6d04164018.png)

 
### (1) DWG to PNG
<img src=https://user-images.githubusercontent.com/61686244/140644084-5d54b60f-ade8-44b0-b9b3-9e3cc5544d55.png width="600" height="350"/>


 
### (2) Dataset-Detection

* Labelimg

 1)
![image](https://user-images.githubusercontent.com/61686244/143766264-5307513d-5179-4185-9431-6c7b242973e3.png)

 2)
![image](https://user-images.githubusercontent.com/61686244/143766291-82a12215-a45a-48e8-9ae3-d8009d60519f.png)

 3)
![image](https://user-images.githubusercontent.com/61686244/143766234-5f08cb38-18ea-4f51-beae-11bcc9aa167f.png)



### (3) Dataset-Recognition
![image](https://user-images.githubusercontent.com/61686244/143766195-4634ac46-366c-4d47-b5bd-019ff00c10f6.png)
![image](https://user-images.githubusercontent.com/61686244/143766198-d6100669-8631-4cce-93fe-39a5f4b98e60.png)





### (4) Train-YOLV4(Detection)



<img src=https://user-images.githubusercontent.com/61686244/140644564-8a0b8e71-82c1-4cc1-82b0-267dbf084091.png width="600" height="350"/>

### (5) Train-STARNet(Recognition)

<img src=https://user-images.githubusercontent.com/61686244/140644576-83318270-68b7-4f2e-87a6-8d9dfe99eddd.png width="600" height="350"/>

### (6) Detection Result


<img src=https://user-images.githubusercontent.com/61686244/140699624-245d2041-9b11-4230-b42a-5ed6e7c164f0.png width="600" height="350"/>

### (7) Recognition 전 전처리

* 검정 pixel 삭제 후 경량 초해상도 네트워크를 이용하여 2배 확대

![전처리](https://user-images.githubusercontent.com/61686244/143766636-04334ec0-567f-4580-8fa5-dcd6153246ce.png)

### (8) Recognition save 


<img src=https://user-images.githubusercontent.com/61686244/140699737-fc51ce5a-f7bf-4a52-9eed-5ce25b6efc36.png width="600" height="350"/>

### Train
<pre>
<code>
$ python train_oversampling.py --single-cls --device '원하는 gpu' (복제augmentation추가원하면 입력->"--srmix --srimx-alpha '원하는횟수' ") 
</code>
</pre>

### Test
<pre>
<code>
$ python detect_edit.py  --device '원하는 gpu' --weights '원하는 학습 pt파일' --save-txt
</code>
</pre>

