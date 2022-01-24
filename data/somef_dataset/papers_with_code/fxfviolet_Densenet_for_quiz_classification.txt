# DenseNet卷积网络对quiz数据集的分类

#### 用DenseNet卷积网络对quiz数据集进行分类，编写densenet卷积网络代码，插入到slim框架中。先用函数train_image_classifier.py对训练集进行训练，得到分类参数，再用函数eval_image_classifier.py对校验集进行校验。

DenseNet卷积网络参考论文： https://arxiv.org/abs/1608.06993

根据论文，编写的DenseNet主要代码如下：   

##### 1) 先定义一个复合函数，包括对输入张量归一化(batch_norm)、激活(relu)、卷积(conv)和dropout。    
    def bn_act_conv_drp(net, num_outputs, kernel_size, scope='block'):
        net = slim.batch_norm(net, scope=scope + '_bn')
        net = tf.nn.relu(net)
        net = slim.conv2d(net, num_outputs, kernel_size, scope=scope + '_conv')
        net = slim.dropout(net, scope=scope + '_dropout')
        return net
    
##### 2) 定义一个层连接函数，调用复合函数bn_act_conv_drp，对输入层进行非线性变换生成特征层tmp，新生成的特征层tmp加入到输入层net，形成新的输入层，经过layers次连接后，最后得到的net作为输出层，传递给densenet函数。layers次连接后输出层net的输出通道数：growth0 + growth x layers，即最开始的net的通道数加上经过layers次连接后的通道数。
    def block(net, layers, growth, scope='block'): 
        for idx in range(layers):
            bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],scope=scope + '_conv1x1' + str(idx)) 
            tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],scope=scope + '_conv3x3' + str(idx))     
            net = tf.concat(axis=3, values=[net, tmp])                      
        return net                     
                 
##### 3) 定义稠密连接函数densenet，先是【7x7】卷积和【3x3】最大池化，之后调用了4个block。前3个block之后，再【1x1】卷积和【2x2】平均池化作为过渡层，第4个block之后，再【7x7】平均池化，每个block的网络层数不同。最后是一个输出通道数为num_classes的全连接层，再用Softmax得到最终分类结果。返回分类结果logits和包含各个卷积后的特征图字典表end_points。 
     def densenet(images,num_classes,growth,scope='densenet'):    
         with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):       
             with slim.arg_scope as ssc:            
                net = slim.conv2d(images, 2*growth, [7, 7], stride=2, scope='pre_conv2')            
                net = slim.max_pool2d(net, [3, 3], stride=2, padding='same',scope='pre_pool2')  
      
                net = block(net, 6, growth, scope='block1')                      
                net = bn_act_conv_drp(net, growth, [1, 1], scope='transition1_conv2')           
                net = slim.avg_pool2d(net, [2, 2], stride=2, padding='same',scope='transition1_pool2')        
                                                         
                net = block(net, 12, growth, scope='block2')      
                net = bn_act_conv_drp(net, growth, [1, 1], scope='transition2_conv2')
                net = slim.avg_pool2d(net, [2, 2], stride=2, padding='same',scope='transition2_pool2')         
             
                net = block(net, 24, growth, scope='block3')
                net = bn_act_conv_drp(net, growth, [1, 1], scope='transition3_conv2')
                net = slim.avg_pool2d(net, [2, 2], stride=2, padding='same',scope='transition3_pool2')      

                net = block(net, 16, growth, scope='block4')      
                net = slim.avg_pool2d(net, net.shape[1:3], scope='global_pool2') 
                                                                        
                net = slim.flatten(net, scope='PreLogitsFlatten')
                logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
                                                                     
                net = slim.flatten(net, scope='PreLogitsFlatten')
                logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
 
                end_points['Logits'] = logits
                end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

        return logits, end_points
