# Tensorflow model pruning: 
## Background
This project was motivated for pruning on Depthwise Separable Convolution. Although the series model of MobileNet has been widely used in edge computing, the models could be through quantization and pruning to achieve a higher speed of inference.

We use structural pruning on the model in TF1.15 to reduce inference time practically.


## Table of contents
1. [Method introduction](#method-introduction)
2. [Pruning-related hyperparameters](#hyperparameters)
3. [Pruning process](#process)
4. [Example](#example)

### Method introduction <a name="method-introduction"></a>
This method now is just support for Inverted-Residual block[[1]](#ref) and Squeeze-and-Excitation block[[2]](#ref). Here use the characteristics of these two blocks, the internal channels of the block are pruned without affecting the output channel size, as shown below.

#### Inverted Residual Block
<table border="0" cellpadding="0" cellspacing="0" style="width: 400;">
<tbody><tr>
<td >　<img alt="" title="Visit Computer Hope" src="https://raw.githubusercontent.com/Gideon0805/Tensorflow_Model_Pruning/main/pic/Inverted_residual_block_nonpruned.png" />Non pruned　</td>
<td>　<img alt="" src="https://raw.githubusercontent.com/Gideon0805/Tensorflow_Model_Pruning/main/pic/Inverted_residual_block_pruned.png" />Pruned　</td>
</tr></tbody></table>

#### Squeeze-and-Excitation Block
<table border="0" cellpadding="0" cellspacing="0" style="width: 400;">
<tbody><tr>
<td >　<img alt="" title="Visit Computer Hope" src="https://raw.githubusercontent.com/Gideon0805/Tensorflow_Model_Pruning/main/pic/SEblock_nonpruned.png" />Non pruned　</td>
<td>　<img alt="" src="https://raw.githubusercontent.com/Gideon0805/Tensorflow_Model_Pruning/main/pic/SEblock_pruned.png" />Pruned　</td>
</tr></tbody></table>

### Pruning-related hyperparameters <a name="hyperparameters"></a>

In our codes, we define pruning represent soft-pruning and strip as a real prune.
The pruning library allows for specification of the following hyper parameters:

|Hyperparameter               | Type    | Default       | Description |
|:----------------------------|:-------:|:-------------:|:--------------|
| input_ckpt_path | string | None | The pretrained model checkpoint path for soft-pruning.|
| scope_names | list of strings | [""] | The scope name is from tf.name_scope or tf.variable_scope, and the scope should be include inverted residual block or SE block. |
| scope_conv_dict   | dictionary | {} | All the convolution node name under the scope_names. |
| conv_vars_dict | dictionary | {} | The weight variable names of convolution. |
| bias_vars_dict | dictionary | {} | The bias or batch-normalization variable name convolution. |
| pruning_filters_dict | dictionary | {} | The decay factor to use for exponential decay of the thresholds |
| pruned_ckpt_path | strings | '' | The checkpoint ready to retrain, after soft-pruning. |
| retrained_ckpt_path| strings | '' | Retrained checkpoint from pruned_ckpt_path. |
| input_tensors_map |dictionary | {} | It will be provided for tf.train.import_meta_graph() to rebuild the model graph. |
| ckpt_to_strip| string | '' | The checkpoint, training nodes have been removed, ready to prune the redundancy filters. |
| strip_nodes_dict | dictionary | {} | The node names of convolution that need to be pruned. |
| strip_vars_dict | dictionary | {} | The variable names that need to be pruned and update the variable value. |
| output_dir | string | None | Output directory. |
| output_nodes | string | 'output' | The output nodes of the model, if there is more than one node, using a comma to separate. For example: 'output1,output2,output3' |
| export_ckpt_path | string | '' | Checkpoint file path. |
| export_pb_path | string | '' | PB file path. This is for inference. |


### Pruning process <a name="process"></a>
First, applying soft-pruning[[3]](#ref) on pretrained-model and retraining until the convolution filters are convergence to zero. (Here, we use the definition of redundancy from Liu et al.[[4]](#ref) to determine the pruning filter.) Then remove the zeros-filter and export it to pb file for inference.

#### Soft-Pruning Process:
1. set_output_dir(self, dir_path=None)
2. get_pruning_params()
3. set_threshold(self, value)
4. get_pruning_filters()
5. get_pruning_ckpt(pruning_filters_path=None)
6. Using the ckpt that return from get_pruning_ckpt() to retrain
7. get_pruning_summary() calculate the failure rate to set retrain to be true or false.
8. pruning_process(retrain)
9. Repeat step2. to step8. until soft-pruning is converged to your requirements.

#### The example of filter's weight values after soft-pruning, as shown below:
<table border="0" cellpadding="0" cellspacing="0" style="width: 400;">
<tbody><tr>
<td>　<img alt="" src="https://raw.githubusercontent.com/Gideon0805/Tensorflow_Model_Pruning/main/pic/Inverted_residual_block_soft-pruning.png" />Inverted Residual block　</td>
<td>　<img alt="" src="https://raw.githubusercontent.com/Gideon0805/Tensorflow_Model_Pruning/main/pic/SEblock_soft-pruning.png" />SE block　</td>
</tr></tbody></table>


#### After retraining of soft-pruning, Structural Pruning Process:
1. get_rebuild_graph
2. get_strip_params()
3. get_strip_ckpt()

#### The example of filter's weight values after structural-pruning, as shown below:
<table border="0" cellpadding="0" cellspacing="0" style="width: 400;">
<tbody><tr>
<td >　<img alt="" src="https://raw.githubusercontent.com/Gideon0805/Tensorflow_Model_Pruning/main/pic/Inverted_residual_block_strip.png" />Inverted Residual block　</td>
<td>　<img alt="" src="https://raw.githubusercontent.com/Gideon0805/Tensorflow_Model_Pruning/main/pic/SEblock_strip.png" />SE block　</td>
</tr></tbody></table>


### Example: <a name="example"></a>

*   [train_pruning_class.py](https://github.com/Gideon0805/Tensorflow_Model_Pruning/tree/main/src/train_pruning_class.py)


```python
# Make a scope_names list
pruning_scopes = ['SE1', 'SE2', 'SE3']
for i in range(1,17):
    name = 'MobilenetV2/expanded_conv_' + str(i) + '/'
    pruning_scopes.append(name)
    pass

output_dir = '/workspace/model_pruning/Testing/Pruning_Class_Test'
# Crate Pruning class
from tf1pruning import Pruning
tf_pruning = Pruning(
    input_ckpt_path=FLAGS.pretrained_model_path,
    scope_names=pruning_scopes,
    output_dir=output_dir
    )

# In training function
# create your own threshold list for soft-pruning
# and use for loop to retrain soft-pruning until convergence
th_steps = [0.8, 0.7, 0.6, 0.55, 0.5]
for th in th_steps:
    try_count = 0
    failed_rate = 1.0
    # Set the soft-pruning threshold.
    tf_pruning.set_threshold(th)
    # Through failed_rate and try_count to determine if the checkpoint needs to retrain again.
    while failed_rate>0.1:
        if try_count == 0:
            pruning_output_dir = os.path.join(output_dir, 'TH'+str(th))
            tf_pruning.set_output_dir(pruning_output_dir)
            ckpt_to_retrain = tf_pruning.pruning_process(retrain=False)
        else:
            pruning_output_dir = os.path.join(output_dir, 'TH'+str(th)+'_Repruned'+str(try_count))
            tf_pruning.set_output_dir(pruning_output_dir)
            ckpt_to_retrain = tf_pruning.pruning_process(retrain=True)
        try_count = try_count + 1

        # We use model_params to specify the checkpoint that need to be retrained.
        model_params['pretrained_model_path'] = ckpt_to_retrain
        # Training with ckpt_to_retrain
        task_graph = tf.Graph()
        with task_graph.as_default():
            global_step = tf.Variable(0, name='global_step', trainable=False)

            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth = True
            ##########################
            # Your own training script
            config = ( tf.estimator.RunConfig().replace(...) ) 

            model = tf.estimator.Estimator(model_fn=model_fn,
                                           model_dir=pruning_output_dir,
                                           config=config,
                                           params=model_params)

            print(
                ('\n validation data number: {} \n').format(
                    len(list(tf.python_io.tf_record_iterator(FLAGS.validationset_path)))
                )
            )

            pip = Pipeline()
            model.train(
                input_fn=lambda: pip.data_pipeline(
                    datasets,
                    params=pipeline_param,
                    batch_size=FLAGS.batch_size
                ),
                steps=FLAGS.training_steps,
                saving_listeners=[
                    EvalCheckpointSaverListener(
                        model,
                        lambda: pip.eval_data_pipeline(
                            FLAGS.validationset_path,
                            params=pipeline_param,
                            batch_size=FLAGS.validation_batch_size
                        )
                    )
                ]
            )
            print('Training Process Finished.')
            ##########################
        #==== Training End ====
        # get retrained ckpt
        retrained_ckpt_path = tf_pruning.get_retrained_ckpt(pruning_output_dir)
        # Analyze how many filters don't converge to zero.
        # If the failed rate is higher than you expect, 
        # do soft-pruning on retrained_ckpt_path and retrain.
        failed_rate = tf_pruning.get_pruning_summary()
        pass

```

## References <a name="ref"></a>
[1] Sandler et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks". (https://arxiv.org/pdf/1801.04381.pdf) <br> 
[2] Hu et al. "Squeeze-and-Excitation Networks". (https://arxiv.org/pdf/1709.01507.pdf) <br>
[3] He et al. "Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks". (https://arxiv.org/pdf/1808.06866.pdf) <br>
[4] Liu et al. "Computation-Performance Optimization of Convolutional Neural Networks with Redundant Kernel Removal". (https://arxiv.org/pdf/1705.10748.pdf) <br>
