# Practical tips for lightweight deep learning

Here I'd like to share some tips after struggling to run a deep learning model on desktop CPUs and mobile phones.

## Use a well-known architecture
Mobilenet V2 is a safe choice. It have been tested in various problems, and you can find implementations and pretrained models for most deep learning frameworks. Also, the operations in Mobilenet are highly optimized in the frameworks. 

You should be careful when you try a new architecture. A new architecture having fewer parameters can be slower when the architecture is not optimized for CPUs and mobile environments. 

## Try different deep learning frameworks
There is no silver bullet working best for all environments in my experiences. Try different frameworks and compare the performance in your target environment. And use the latest version of the frameworks.

When you try different frameworks, train the model again in the frameworks. There are model conversion tools like ONNX, but they are not optimized yet. 

## Turn on all optimization options in the frameworks
Some optimization options might be disabled in the precompiled pip versions. Try to enable all optimizations to get maximum performances. You have to compile the frameworks by yourself.

## Use GPU if it is available
GPU is definitely faster than CPU. It is true for mobile phones. TF-lite [https://www.tensorflow.org/lite/performance/gpu] supports mobile GPUs.

## Use quantization
Weight quantization is a good way to reduces inference latency without much loss in accuracy. See links below. 

TF lite [https://www.tensorflow.org/lite/performance/post_training_quantization]  
QNNPACK for Caffe2 [https://code.fb.com/ml-applications/qnnpack/]

Be aware that the operations in your network should have implementations for quantization. 

## Profiling
It helps you to find out bottlenecks in your model. 

Tensorflow [http://deeplearnphysics.org/Blog/2018-09-25-Profiling-Tensorflow.html]  
Pytorch [https://varblog.org/blog/2018/05/24/profiling-and-optimizing-machine-learning-model-training-with-pytorch/]

## Fine-tune the size of the model
Reduce the input resolution, depth, or width of the network. Recent paper EfficientNet gives great insights on doing this. [https://arxiv.org/pdf/1905.11946.pdf]

---

Pull requests and comments are always welcome
