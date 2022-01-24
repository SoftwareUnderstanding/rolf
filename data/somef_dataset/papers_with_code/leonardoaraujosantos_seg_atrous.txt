# Test segmentation with Atrous Convolution

### Output size
Output size formula:

o = [i + 2*p - k - (k-1)*(d-1)]/s + 1 

Where,
* i: input size
* p: pad
* k: Kernel size
* d: dilation factor
* s: stride

#### References
* https://www.youtube.com/watch?v=t2L9VnGi7hA
* https://arxiv.org/pdf/1709.00179.pdf
* https://ezyang.github.io/convolution-visualizer/index.html
* https://arxiv.org/pdf/1706.05587.pdf
* https://blog.exxactcorp.com/atrous-convolutions-u-net-architectures-for-deep-learning-a-brief-history/
* https://arxiv.org/pdf/1706.05587.pdf
* http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review
* https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5
