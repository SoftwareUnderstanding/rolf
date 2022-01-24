# Who Judges Books by Their Covers?
## Technical Supplement
This is the technical supplement to an analysis conducted separately. In this archive you will find the code (in Jupyter notebooks) that was used to generate the analysis, and in this readme file, I will give a deeper explanation as to precisely what was done, in addition to some caveats to the results. If you have any questions regarding this analysis, please contact me.

## References and Resources
* The Data used for the Analysis was pulled from [zygmuntz/goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k)
* There are a number of technical references for VAEs, of particular note are: 
    * Christopher P Burgess et al. “Understanding disentangling in β -VAE”. In: Nips 2017 Nips (2017). arXiv: 1804.03599.
    * Irina Higgins et al. “beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework”. In: Iclr July (2017), pp. 1–13. url: https://openreview.net/forum?id=Sy2fzU9gl.
    * Diederik P Kingma and Max Welling. “Auto-Encoding Variational Bayes”. In: Ml (2013), pp. 1–14. issn: 1312.6114v10. doi: 10.1051/0004- 6361/201527329. arXiv: 1312.6114. url: http://arxiv.org/abs/1312.6114.
* For a brief introduction to developing standard VAEs, I suggest [this](http://kvfrans.com/variational-autoencoders-explained/) blog post, walking through the development in Python.
* There are an incredible number of tutorials explaining SVMs, and their implementation across various lanaguages. [This](https://blog.statsbot.co/support-vector-machines-tutorial-c1618e635e93) is one that does a good visual job of justifying the use.

## Qualifications on the Findings
The major qualification on the findings here have to do with claiming that all of the information used comes directly from the book covers. While it is true that the only data used in the modelling process was the projected images, it is seldom the case that data carries with it no further information. In particular, the way that SVMs are fit, the representative nature of the training data contains valuable information for the model - namely that a user is more likely to select one rating versus another. As such, you can use an informed baseline where instead of selecting a random prediction with equal probabilities, you select a a random prediction given the individuals previous selection patterns. This will certainly outperform the random selection, though, there is still added benefit to the information contained in the covers. 

This is a worthwhile lesson to keep in mind for all models that are correctly constructed. In fact, one of the great strengths of modelling in particular ways in the implicit ability to account for the prior distribution of the data.

## Navigating the Repository
The ```Exploration Notebook.ipynb``` contains my initial building of the VAE and the exploration of the data briefly. There are comments to let you know what the cells are doing, and looking through it showcases my initial programatic exploration of the idea to use a VAE to encode the covers.

The ```User Analysis.ipynb``` contains the actual user-level analysis that is reported in the write-up. The actual SVMs are fit, and the independent analysis is run, based on the users particular ratings. 