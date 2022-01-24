ISA525700 Computer Vision for Visual Effects<br/>Homework 1 (Color-Transfer and Texture-Transfer)
===

## [Cycle GAN](https://arxiv.org/pdf/1703.10593.pdf)
### Introduction
Cycle gan跟傳統的gan做圖像轉換的方式不同，它不需要配對的數據集(paired image data set)；利用兩個generator、discrimnator和轉換的一致性(consistency)，cycle gan只需要不同風格的unpaired image data set即可運作。

### Architecture
Cycle gan總共有兩個Genertoar(G, F)，分別把圖像從Domain X到Domain Y，跟反向；同時也有兩個相對應的Discriminator(DY, DX)。

不過，因為input為未配對數據集，因此在風格轉換後(X -> Y')，還需要將結果再逆轉換回來(Y' -> X')，並做比較以確保轉換過後結構相似。

![](https://i.imgur.com/vqnmh1F.png)

### Loss Function
```
L(G, F, DX, DY) = Lgan(G, DY, X, Y) + Lgan(F, DX, Y, X) +　λLcyc(G, F) 
```
_Lcyc(G, F) = Ex∼     pdata(x)[‖F(G(x))−x‖1] + Ey∼pdata(y)[‖G(F(y))−y‖1], whcih represents loss of structure cmparison_

### Training Process
![](https://i.imgur.com/9UhfBha.png)


### Inference Cycle GAN in personal image
#### orange2apple


result v.s. original image


![](https://i.imgur.com/CFhqolS.png)

![](https://i.imgur.com/r7Wg1lO.png)

#### apple2orange


result v.s. original image


![](https://i.imgur.com/vwa18ZX.png)

![](https://i.imgur.com/hQtVrfa.png)

![](https://i.imgur.com/uAqtVHE.png)

#### 

## Other Methods
### [Color Transfer Between Images](https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf)

#### Introduction
該論文利用簡單的統計分析，將一張圖的顏色特徵轉移到另外一張圖上，其中，色彩校正的部分主要是藉由選擇合適的source image，並將其特徵應用到target image上來實現。

#### Algorithm
1. 將source image和target image由RGB轉為LAB
2. 計算source image和target image在LAB color space三個channel的標準差和平均值
3. 將source image的pixel值減去平均pixel值，再除以標準差，之後加上target image的平均值，得到最後的LAB值
4. 從LAB轉換為RGB

#### Concept
RGB三個channel具有相當強的關聯性，而做color transfer的同時，適當地改變三個channel比較困難，因此選擇利用三個channel互不相關的LAB color space來進行計算，過程中，把整個分佈轉換，使LAB color space的每個維度都是原本RGB 值的線性組合，比較會顧及相似顏色間的相對關係。

#### Example
|![](https://i.imgur.com/RrztFYY.jpg)|![](https://i.imgur.com/APtBT2Q.jpg)|![](https://i.imgur.com/j4ttZKo.jpg)|
| ----------------- | --------------- | --------------- |

### [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)
#### Introduction

該論文透過卷積神經網路，將圖片的內容及風格分開並重建，提供一個style transfer的做法。
#### Architecture
![](https://i.imgur.com/kuZTYgs.png)
##### Concept
利用pre-trained VGG19 model提取圖片不同layer的特徵，作為圖像和風格的特徵。
- Content:在network中，較低層級保留較多的原始圖像，而較高層級中雖然沒有細節的pixel但保留了high level的內容feature，因此可以利用高層的特徵來對圖像做recontruct。
- Style:在原本的CNN架構上建立新的feature space來提取風格的特徵。作法為計算不同layer所產生的不同feature之間的correlations。在reconstruction中，該結果確實獲得的原始影像的texture、color等特徵。

#### Loss Function
- Loss Function = content loss + style loss
- Content loss : 原圖與預測圖的content feature差距
- Style loss : 原圖與預測圖的style feature差距

#### Flow
![](https://i.imgur.com/zVpm139.png)
$G^{l}_{ij}$ is the inner product between the vectorised feature maps of the initial image $i$ and $j$ in layer $l$,
$w_{l}$ is the weight of each style layers
$A_l$ is that of the style image
$F_l$ is layer-wise content features of the initial image
$P_l$ is that of the content image
* $\alpha$ and $\beta$ is the content weight and   style weight, respectively that controls the weighting factors for content and style reconstruction.

#### Example
![](https://i.imgur.com/felJKFp.jpg)

![](https://i.imgur.com/FkezUlt.png)



### [Universal Style Transfer](https://arxiv.org/pdf/1705.08086.pdf)
#### Introduction
該論文提出了一個通用的reconstruction network，希望能夠對輸入的任意style進行transfer，而不需要重新train model；換句話說，就是希望能夠使用任意的reference image來進行style transfer，擺脫傳統的style transfer對於style和content loss需要通過對layer的嘗試參數，來得到一個和style較爲匹配的表述纔能有較好的效果，且針對不同的style這一步驟需要重新training這樣的缺點。
該論文提出了Whitening & Coloring transform layer (WCT layer)，它的實作觀念在於，對於任何一種style image(reference image)，要能夠使content表現出style的風格，只需在feature map上分布表徵一致。
首先，將feature map減去平均值，然後乘上對自己的協方差矩陣的逆矩陣，來進行whitening的動作，以利將feature map拉到一個白話的分布空間。然後透過對reference image取得feature map的coloring協方差矩陣的方式，將其乘以content image whitening後的結果，並加上平均值，就可以將content image whitening後的feature map空間轉移到reference image圖片上平均分布；最後，透過Stylization Weight Control 的公式：

<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{f_{cs}}&space;=&space;\alpha&space;\widehat{f_{cs}}&space;&plus;&space;(1&space;-&space;\alpha)\widehat{f_c}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{f_{cs}}&space;=&space;\alpha&space;\widehat{f_{cs}}&space;&plus;&space;(1&space;-&space;\alpha)\widehat{f_c}" title="\widehat{f_{cs}} = \alpha \widehat{f_{cs}} + (1 - \alpha)\widehat{f_c}" /></a>

就可以完成將reference image整合input image的動作。

#### Pipeline
![](https://i.imgur.com/V6eutky.png)

#### Loss Function
<a href="https://www.codecogs.com/eqnedit.php?latex=L&space;=&space;\left&space;\|&space;I_{0}&space;-&space;I_{i}\right&space;\|_{a}^{b}&space;&plus;&space;\lambda&space;\left&space;\|&space;\Phi&space;(I_{0})&space;-&space;\Phi&space;(I_{i})\right&space;\|_{a}^{b}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L&space;=&space;\left&space;\|&space;I_{0}&space;-&space;I_{i}\right&space;\|_{a}^{b}&space;&plus;&space;\lambda&space;\left&space;\|&space;\Phi&space;(I_{0})&space;-&space;\Phi&space;(I_{i})\right&space;\|_{a}^{b}" title="L = \left \| I_{0} - I_{i}\right \|_{a}^{b} + \lambda \left \| \Phi (I_{0}) - \Phi (I_{i})\right \|_{a}^{b}" /></a>

#### Example
![](https://i.imgur.com/0n5p0oR.png)


### Comparisons 
#### photo2monet
| content  | style    | Cycle GAN (with monet style)| Neural Algorithm|  Universal| color transfer| 
| ----------------- | --------------- |--------------- | --------------- |--------------- | --------------- | 
|![](https://i.imgur.com/QKJGL2S.jpg) |![](https://i.imgur.com/DfxqM53.jpg)|![](https://i.imgur.com/3QEVtf2.png)|![](https://i.imgur.com/SkBQ9Ld.jpg)|![](https://i.imgur.com/JCvdIM1.jpg)|![](https://i.imgur.com/2BfmXFP.jpg)|
|![](https://i.imgur.com/yY5xJ4I.jpg)|![](https://i.imgur.com/DfxqM53.jpg)|![](https://i.imgur.com/CqO1e7S.png)|![](https://i.imgur.com/oWwDxsn.jpg)|![](https://i.imgur.com/HYbzKdI.jpg)|![](https://i.imgur.com/PRdClBu.jpg)|
|![](https://i.imgur.com/CsHBjda.jpg)|![](https://i.imgur.com/DfxqM53.jpg)|![](https://i.imgur.com/mfBttGE.png)|![](https://i.imgur.com/vQlLvVi.jpg)|![](https://i.imgur.com/pH2wO1k.jpg)|![](https://i.imgur.com/R0U2VHf.jpg)|
|![](https://i.imgur.com/xSwyzfy.jpg)|![](https://i.imgur.com/DfxqM53.jpg)|![](https://i.imgur.com/wyaTbrS.png)|![](https://i.imgur.com/fSnrbKI.jpg)|![](https://i.imgur.com/ENnqJrJ.jpg)|![](https://i.imgur.com/t3VX7rv.jpg)|
|![](https://i.imgur.com/QKJGL2S.jpg)|![](https://i.imgur.com/MqSFc8o.jpg)|![](https://i.imgur.com/3QEVtf2.png)|![](https://i.imgur.com/QyUaFj9.jpg)|![](https://i.imgur.com/HxEPSTj.jpg)|![](https://i.imgur.com/5LundSa.jpg)|
|![](https://i.imgur.com/yY5xJ4I.jpg)|![](https://i.imgur.com/MqSFc8o.jpg)|![](https://i.imgur.com/CqO1e7S.png)|![](https://i.imgur.com/Cv0qqo4.jpg)|![](https://i.imgur.com/FRbpL1q.jpg)|![](https://i.imgur.com/BCB7kWi.jpg)|
|![](https://i.imgur.com/CsHBjda.jpg)|![](https://i.imgur.com/MqSFc8o.jpg)|![](https://i.imgur.com/mfBttGE.png)|![](https://i.imgur.com/RwD8Ydx.jpg)|![](https://i.imgur.com/2WomYI2.jpg)|![](https://i.imgur.com/t3Xn9MP.jpg)|
|![](https://i.imgur.com/DgAhzHT.jpg)|![](https://i.imgur.com/MqSFc8o.jpg)|![](https://i.imgur.com/wyaTbrS.png)|![](https://i.imgur.com/VvTOuo1.jpg)|![](https://i.imgur.com/dxNVGAH.jpg)|![](https://i.imgur.com/iQYs7er.jpg)|


#### Features

| Method    |  Aspect  |  Algorithm   | Input | Output
| ----------------- | --------------- | --------------- | --------------- | ---------------
| Cycle GAN  | color transfer, style transfer...   | Generative and Discriminative model   | two sets of unpaired image     | general style  image
| Neural | style transfer  | neural network   |   Content Image v.s. Style Image  | single-image-style image
| Universal | style transfer   | neural network   | Content Image v.s. Style Image      | single-image-style image
 Color Transfer| Color Transfer| Statistical Transform| source image v.s. target image| single-image-color image

#### Conclusion
- Traditional:
    - Color transfer: 主要是利用數學的方法將target image線性轉換為與source image相似的色彩主調，較能針對依賴色彩的風格，而不適用於筆觸、花紋等其他特徵。
- GAN ：
    - Cycle GAN: 在架構上，cycle GAN所學習到的為如何產生指定的某一種texture,style的圖片（from training data)，其方法通過unpaired image經過GEN和DIS來反覆更新達成，這種架構除了能處理style外，在其他case上也能有好的結果。
- CNN model : 
    - Neural Style Transfer : 透過CNN model將圖片的兩種feature的提取進行reconstruction，所針對的為某一張圖的風格，並特過content、style factor來決定style 比重來完成風格轉換。
    - Universal Style Transfer : 在一個content layer中使用whitening and coloring transforms，同時增加一個pre-trained的general encoder-decoder network，使同一個model可以應用在不同style，達到Universal的效果。 


