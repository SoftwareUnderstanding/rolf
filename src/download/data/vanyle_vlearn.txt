# VLearn

VLearn is a C++ library containing all sort of stuff out of the box to make cool projects machine learning projects.
Every aspect of VLearn is separated in a different folder.

While VLearn contains machine learning features, it also has methods for file manipulation, networking and serialization,
to help you build awesome deep learning projects.


VLearn is the framework used for the Distributed Machine Learning project.

# Performance

Our goal is to be faster than tensorflow. No more waiting for days to train your model.
To do this, we provide feature to distribute the computation across devices and we support more GPUs.
We plan to have a CUDA and an OpenCL backend.

# Building and Usage.

VLearn compiles to a static library (vlearn.a) that you can then link against.
When linking vlearn, you should also link with dbghelp on windows to access the debugging features.

VLearn is compiled with the vapm build system and the MinGW compiler. VLearn aims to be compatible with Windows and Linux.
I won't add compatibility with the Microsoft Visual C++ Compiler but I will accept merge requests adding this compatibility.

# Usage

Examples of how to use VLearn are available in the `test` folder.
Here is how to train a network to learn a linear operation:

```cpp
#include <... put all the includes ...>

using namespace vio;

void main(){
	// Define the network
	NeuralNetwork nn;
	DenseLayer l1(3,4); // input size: 3 floats
	DenseLayer l2(4,1); // output size: 1 float
	l1.randomInit(5);
	l2.randomInit(5);
	nn.layers.push_back(&l1);
	nn.layers.push_back(&l2);

	nn.prepare();  // compile the network.

	// let's generate some data to train the network !
	// We try to teach the network a simple linear function.
	// You can change this to try to make it learn various stuff.
	std::vector<Vector> trainingInputs;
	std::vector<Vector> trainingOutputs;
	for(u32 i = 0;i < 1000;i++){
		Vector newIn(3);
		Vector newOut(1);

		newIn.at(0) = randomFloat()*20 - 10;
		newIn.at(1) = randomFloat()*20 - 10;
		newIn.at(2) = randomFloat()*20 - 10;

		newOut.at(0) = newIn.get(0) * 3 + newIn.get(1) * 5 + newIn.get(2) * 0;

		trainingInputs.push_back(std::move(newIn));
		trainingOutputs.push_back(std::move(newOut));
	}
	
	for(u32 i = 0;i < 1000;i++){
		nn.train(trainingInputs,trainingOutputs,0.001); // 0.001 is optimal, it's the learning rate
	}
	float e = nn.loss(trainingInputs,trainingOutputs);
	debug("Loss : %f",e);

	// Test the network:
	Vector my_input(3);
	my_input.at(0) = 3;
	my_input.at(1) = 5;
	my_input.at(2) = -3;
	Vector my_output = nn.apply(my_input); // evaluate the network on the input
	my_output.print();
}
```

# Organisation of the project and information for contributing

## Machine learning

The machine learning related code is in the ml folder.
We provide code to build and train a neural network, with methods similar to the keras ones.
We provide various Layer types, Optimizers (I mean, we have adam, you won't need anything else) and computing backends.

## File manipulation

VToolbox has some methods to make reading / writing various file types easier.
This includes images, audio and text files.
The methods are similar to the ones in Python.
Check out the documentation (`doc/index.html`) for more info.

## Debugging

VCrash is a part of VToolbox, checkout vcrash here: https://github.com/vanyle/vcrash
VCrash allows us to print stack traces and causes of crash even when no debugger is attached to the program.

## Documentation

Checkout `/doc/` for detailed documentation of every package.
`/doc/` contains an `index.html` with auto-generated searchable documentation for every package.
For examples, see the `/test/` folder, it contains working code examples of most features of vtoolbox.

## TODO

- Implement more computing backends (GPUs)
- Implement for optimizer
- Implement non-gradient descent based learning systems (evolution, etc...)
- Implement "layer branches"
- Implement distributed training over the network

# References

We used the following lecture to implement gradient descent:
https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf

We used the adam paper to implement adam:
https://arxiv.org/abs/1412.6980

We used the following paper as a parallelisation reference for neural networks:
https://papers.nips.cc/paper/2006/file/77ee3bc58ce560b86c2b59363281e914-Paper.pdf