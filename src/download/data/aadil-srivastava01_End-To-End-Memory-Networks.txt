# End-To-End-Memory-Networks
Neural network with a recurrent attention model over a possibly large external memory

The notebooks in the repository implements the idea of End-To-End Memory Network which was presented in the paper http://arxiv.org/abs/1503.08895.

The architecture is a form of Memory Network
but unlike the model in that work, it is trained end-to-end, and hence requires
significantly less supervision during training, making it more generally applicable
in realistic settings.

A number of recent efforts have explored ways to capture long-term structure within sequences
using RNNs or LSTM-based models. The memory in these models is the state
of the network, which is latent and inherently unstable over long timescales. The LSTM-based
models address this through local memory cells which lock in the network state from the past. In
practice, the performance gains over carefully trained RNNs are modest.
This model differs from these in that it uses a global memory, with shared read and write functions.
However, with layer-wise weight tying this model can be viewed as a form of RNN which only
produces an output after a fixed number of time steps (corresponding to the number of hops), with
the intermediary steps involving memory input/output operations that update the internal state.

Basic Architecture of the network is as follows:

![Alt text](https://cdn-images-1.medium.com/max/800/1*mapbkOsuwaA0sm4-M_HSOg.jpeg "N-N Memory Networks")

Kindly find the paper for more details.



