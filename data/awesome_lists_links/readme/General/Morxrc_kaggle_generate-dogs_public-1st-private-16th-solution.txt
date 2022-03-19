

# kaggle generate dogs 1st-public 16th-private solution

**solution for the kaggles competition**:https://www.kaggle.com/c/generative-dog-images

[@MeisterMorxrc](https://www.kaggle.com/meistermorxrc)

## 生成小狗效果展示

![生成小狗效果展示](https://github.com/Morxrc/kaggle_generate-dogs_public-1st-private-16th-solution/blob/master/Generate%20dogs%20pic.png)



## 1. competition introduction:

1.Description:  A generative adversarial network (GAN) is a class of machine learning system invented by Ian Goodfellow in 2014. Two neural networks compete with each other in a game. Given a training set, this technique learns to generate new data with the same statistics as the training set.

In this competition, you’ll be training generative models to create images of dogs. Only this time… there’s no ground truth data for you to predict. Here, you’ll submit the images and be scored based on how well those images are classified as dogs from pre-trained neural networks. Take these images, for example. Can you tell which are real vs. generated?



2.Evaluation:MiFID

用生成模型所常用的指标[FID](https://baijiahao.baidu.com/s?id=1647349368499780367&wfr=spider&for=pc),之前的Mi是Memorization-informed的简称,官方解释如下:The memorization distance is defined as the minimum cosine distance of all training samples in the feature space, averaged across all user generated image samples. This distance is thresholded, and it's assigned to 1.0 if the distance exceeds a pre-defined epsilon.通俗来说就是即一个衡量你生成图片和原始图片的distance 的惩罚系数(防止你不做训练直接将原图提交上去生成"极为逼真"的小狗)。



## 2. 方案介绍:

Model: DCGAN

介绍：**DCGAN**虽然有很好的架构，但是对**GAN**训练稳定性来说是治标不治本，没有从根本上解决问题，而且训练的时候仍需要小心的平衡**G,D**的训练进程，往往是训练一个多次，训练另一个一次。而对于比赛来说，往往是一锤子买卖，因此为了巩固**GAN**的稳定性，我们做了非常多的工作。

*****

Ps： 一些碎碎念:(根据结果来看,bigGAN 在比赛中大放异彩,因为其本身就是**将正交正则化的思想引入 GAN，通过对输入先验分布 z 的适时截断，不仅极大的减少了GAN的训练时间，而且还大大提升了 GAN 的生成性能和稳定性**,让人不禁感慨这就如他的论文介绍一般:**当代最强GAN**,而我们虽然已经发现了我们的努力并不能逾越DCGAN和BigGAN本身的模型性能，但迫于时间原因，只能硬着头皮继续去做DCGAN的改良,最终结果也表明，这个DCGAN的方案远远超过了其他选手的DCGAN方案得分,因此我觉得也在一定意义上存在一些借鉴价值)。

****

### 方案summary和创新点：

1. 图片的多种预处理(Data Aug):4种

   由于图像噪声较多，例如很多图像的有很多杂物,or人,或者多只狗，有些狗只存在于角落，或者狗的身子特别长之类的，因此在这个比赛中,如何选取合适的剪切方式是一个重点,即如何对狗的位置进行追踪，还有最大程度的将狗头🐶剪切到图像中。

2. 对Generator 的参数在训练后半段做滑动平均（通俗来说类似于BN的均值方差）。https://arxiv.org/abs/1803.05407 SWA

   1. 优点:
      1. 不依赖学习率的变化，设置恒定学习率的时候依然可以达到很好的收敛效果
      2. 收敛速度非常快，平均振幅也非常小。

3. split bn 操作。（对real label 和fake label 一起cat起来做forward,但是对bn分开计算）

   1. 优点:
      1. 此操作可以**显著**起到提速的效果。

4. 修改loss为一种margin loss的方法:

   1. 我们观察到一个batch图片过多的时候,每到训练中后期，真实图片的得分几乎全为1，过拟合非常严重，因此我们让real 得分超过某一个margin loss时,对loss直接置0,从而有效的起到了防止过拟合的作用,训练效果提升非常明显。

5. 其他通用trick:

   1. 对于Generator 进行加深加宽处理.(**注意的是**,对于G 进行加宽处理时,D要与其同步加宽,否则效果会崩)。





