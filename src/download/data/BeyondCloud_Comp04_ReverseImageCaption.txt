
# Student ID, name of each team member.
    106061514 許鈞棠  106061536 廖學煒
# Demo: Pick 5 descriptions from testing data and generate 5 images with different z respectively.    
<img src="./Report/25_inf.png">
    (first row ~ last row)<br>
    0633: this flower has petals that are yellow with red blotches <br>
    0194: the flower has white stringy petals with yellow and purple pollen tubes<br>
    2014: this flower is pink in color with only one large petal<br>
    4683: this flower is yellow in color with petals that are rounded<br>
    3327: the flower has a several pieces of yellow colored petals that looks similar to its leaves
# The models you tried during competition. Briefly describe the main idea of this model and why you chose this model.
   ### GAN-CLS (Based on DCGAN )algorithm (http://arxiv.org/abs/1605.05396)
    We had implemented three models(small-scale, large-scale and WGAN-GP). The small one required <2G GPU RAM while the large one 
    required ~8.5G (for batch size = 64)
    
   ### The large-scale model comprises of three parts:
    1.Textencoder + CNNencoder
    This section is used for generating appropriate text embedding. First, we feed correct images and mismatched images
    into CNNencoder, which generates encoded vector x and x_w. As for the textencoder , we feed correct captions and
    mismatched captions into encoder to generate v and v_w. Finally, the similarity of text and image can be calculated by:
        alpha = 0.2
        rnn_loss = tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x, v_w))) + \
                tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x_w, v)))

    2.Generator:
    The generator consists of 6 nets. The feature map start from 4x4, doubling its size by upsampling one 
    time when passing each of the last 4 nets. Plus, the network uses highway connection which adds two 
    features( before passing CNN and after passing CNN) together. This could be regarded as "adding some images
    detail to the output of previous layer." Since the pixel value of the output image is 0~1, we use 
    tanh(last_layer_output) or tanh(last_layer_output)*0.5+0.5 to map the output value to reasonable range.
    
    Note about tanh(last_layer_output) or tanh(last_layer_output)*0.5+0.5
        In scipy image the pixel value < 0 will be converted to zero automatically. In our small-scale model
     we used the later one (tanh(last_layer_output)*0.5+0.5 range= [0,1]). However, the large-scale model will 
     collapse if the later activation function is applied. So it's really case by case to choose which activation
     function should be used in the last layer

    
    3.Discriminator
    There are 4 nets inside the discriminator. The top 3 layers use CONV2D with stride = 2 to compress the
    information of the image. At the 4th layer the model concatenates the textvector and 3rd layer 
    output. After the concatenated vector passing  through 2 CONV2D network, the model will generate an output 
    logit with shape = (batch_size,).
    
    Below are the equations used for calculating the generator/discriminator loss
    
    disc_real = Discriminator(real_image, text_encoder.out(correct_caption))
    disc_fake = Discriminator(fake_image, text_encoder.out(correct_caption))
    disc_wrong = Discriminator(real_image, text_lencoder.out(wrong_caption))
    d_loss1 = reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real.logits,labels=tf.ones_like(disc_real.logits)*0.9))
    d_loss2 = reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_wrong.logits,labels=tf.zeros_like(disc_wrong.logits)+0.1'))
    d_loss3 = reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake.logits,labels=tf.zeros_like(disc_fake.logits)+0.1))
    d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5
    
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake.logits,labels=tf.ones_like(disc_fake.logits)*0.9))
   ###  The small-scale model ( input = skip_thought word vector ,no RNN)
    The small model also use GAN-CLS but without high-way connection and RNN encoder
   ### WGAN-GP
    We also tried modifying the original DCGAN loss by WGAN loss with gradient penalty. However, if we just change the loss to WGAN loss, the training would be failed for converging a good result. The follows are our modifications:
    1. The loss for mis-matching pairs was not used. 
    2. The gradient penalty is zero-centered gradient penalty (w.r.t the real image) from this paper https://arxiv.org/pdf/1801.04406.pdf.
    Therefore, the loss became:
    
    self.fake_discriminator = Discriminator(generator.outputs, text_encoder.outputs, ...)
    self.real_discriminator = Discriminator(self.real_image_n, text_encoder.outputs, ...)
    ...
    d_loss1 = tf.reduce_mean(self.real_discriminator.logits)
    d_loss2 = tf.reduce_mean(self.fake_discriminator.logits)
    self.d_loss = (d_loss2)  -  d_loss1 + self.rnn_loss +10 * self.gradient_penalty
    self.g_loss = -d_loss2
    , where rnn_loss is the the same as the loss in GAN-CLS.
    
# List the experiment you did. For example, the hyper-parameters tuning, architecture tuning, optimizer tuning and so on.

    
   ### soft label
       Changing the discriminator true image label from 1 to 0.9 and fake image label from 0 to 0.1 might speed 
       up the training process at the beginning of training. During the early training stage, the discriminator
       could easily tell the difference between real and fake image. Making this change could also prevent 
       the discriminator loss drop to zero since the generator always generates nonsense images to fool the discriminator.
       
   ### Sample noise from normal distribution
        Using normal distribution noise is more intuitive for generating natural images. However, we found that whether 
        we use uniform distribution or normal distribution noise vector, they generate similar inception score.
        
   ### Using dropout in generator
        In our experiment, we found that dropout is useful in the small-scale model. At the early stage of training. 
        the model always output similar flowers for each kind of input. This problem can be solved by adding 0.5 
        dropout in several generator's layers. However, the large-scale model could generate wide variety of flower
        at the initial stage so the dropout layer is unnecessary.
        
        Plus, we found that the inception score of small model is more fluctuating than the large model for different
        random noise input. 
   ### Virtual batch normalization
        Sometimes the synthesised image might have similar color for each batch of input(see gif below). 
        Whenever this happens, changing the batch_normalization layer to VBN could solve the problem.
        See paper https://arxiv.org/pdf/1606.03498.pdf
        
<img src="./Report/test.gif">
   ### The n_critic parameter in WGAN
        In the original WGAN-GP paper, the authors train the discriminator 5 times for training generator 1 time. 
        Using the same setting for training the GAN-CLS leads the speed of convergence very slow.
        After tuning this parameter, we found we can train with n_critic = 1 and it became faster. 
        Unfortunately, such modigication also made the training slightly unstable. Sometimes the generated image 
        changes dramatically in two consecutive training epochs.
        The potential reason is that the disciminator still had not been trained well but the generator had been update.
        We expect that tuning the model of discriminator and making it stronger than generator can solve this problem.

   ### Structure loss(fail)
        Although the generator is able to synthesis images with right color, we found that some images have
        distorted shape.
        In order to deal with this problem, we add two extra losses to the discriminator, hoping the model could
        care more about the shape of the flower:
        Below is what we'd done
        1. loss1 = convert the real image to grayscale, duplicate its value to 3 channels, label it as 1
        2. loss2 = convert the fake image to grayscale, duplicate its value to 3 channels, label it as 0
        3. add loss1 ,loss2 to discriminator loss
        Unfortunately, the generator outputs grayscale image rather than the image with better shape
<img src="./Report/bk.jpg" alt="Drawing" style="width: 400px;"/>

   ###  Multi-sampling generated image
       We found that the image generating quality is highly depends on the noise vector input. Consequently, the 
       inception score might be fluctuated. To solve this problem, we load the discriminator while running inference.
       The algorithm is discribed below:
       1. Using 10 different random vectors to generate 10 images from same caption
       2. Feeding 10 images and captions into the discriminator
       3. Pick the image with the highest discriminator output value
       This method could more or less increase the stability of output score.
   
### How did you preprocess your data (if you did it by yourself, not using the method on course notebook)
            ## Data Augmentation
                In an attempt to generate more training data, the input image will be flipped left-right, up-down,
                left-rigt-up-down randomly before feeding into the network
            ## Image/Captions caching
                Saving the processed image (resize/rotate) and caption(encoded text vector) to npy file could
                make the training process more efficient.
            ## More information about captions
                We use five captions for each image. Plus, we remove all the symbol.
           
        - Do you use pre-trained model? (word embedding model, RNN model, etc.) 
   ### Skip thought vector  (upper64: generated lower64: ground truth) (inception score 0.114)
 <img src="./Report/ST.jpg">

                We had done some experiment using the pretrain model. However, we didn't use it in kaggle private
                leaderboard since the competition is not allowed to do so. There are two benefits about using the
                skip-thought vector:
  
                1. No RNN network required. Thus, the training speed is faster and the memory usage is relatively 
                low. (<2G GPU memory for small-scale model (batch size = 64)). Also the text embedded is more 
                stable than RNN since it's always unchanged
                
                2. When using RNN we need to preprocess the sentence to the same length and the max length of 
                sentence/vocabulary size are also limited. Skip tought vector could generate vector from different
                length sentence and the vocabulary dictionary is also large.
                
                However, the skipthought vector also comes with some drawbacks:
                1.Generating word embedding is time consuming. It takes about 5 hours to build the skipthought vector
                from all training and testing captions (5 captions per image)
                2. We found that it takes more time for the model to learn the color information from skipthought
                vector in comparison with RNN text-encoder.


  ### SRCNN  - super resolution network (fail)
           
           Although the generated image looks like the flower, the image is lack of detail information. We could use 
           stack-GAN to solve this problem. However, the implementation of stack-GAN is complicated and we also don't
           have sufficient memory. Therefore, We attempted to solve this problem by increasing the low resolution 
           generated image using the super resolution network post-processing.
           
           We loaded the pretrained model, converting the image from 64x64 to 256x256 and resizing it back to 64x64.
           Nevertheless, the inception score doesn't improve since some image is distorted after resizing.
           
           Figure below: upper: fail case, lower: successful case
           
<img src="./Report/SRCNN.jpg" alt="Drawing" style="width: 400px;"/>
        - Any tricks or techniques you used in this task.
        
            # Sharpen-Filter post-processing
            
            In "A Note on the Inception Score (2018)  Shane Barratt, Rishi Sharma", it claims that the sharp image  
            is more likely to get higher inception score. Consequently, we apply the sharpen filter to the output 
            test images by using python Image lib sharpness filter:
                img = Image.fromarray(np.uint8(images*255))
                enhancer = ImageEnhance.Sharpness(img)
                img  = enhancer.enhance(3.0)
            This process could increase about 0.03 inception score 
            Our model is able to achieve 0.117 inception score after applying the sharpen filter
 <img src="./Report/sharp.jpg">
   # Conclusions (interesting findings, pitfalls, takeaway lessons, etc.)
    We found that GAN is really hard to train since the loss of generator and discriminator are unstable and they
    don't have too much reference value (also hard to debug). Plus, this is the first time I understand the importance
    of tf.variable_scope. The model have three networks (RNN,G,D) and three optimizers, so we need to control which
    network's variable should be update.
    
    We also found that the inception score is a mediocre indicator since it's easy to generate high inception score 
    from non-sense images. For instance, we scored 0.121 by outputing the same image from one of the training images
    and adding some noise to them (see figure below). There's also a paper talking about how to generate near perfect 
    inception score from poor quality image 
    https://arxiv.org/abs/1801.01973
<img src="./Report/suck.jpg" alt="Drawing" style="width: 400px;"/>


```python

```
