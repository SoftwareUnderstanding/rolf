# Text-Anlaytics
Feature Extraction, Sentiment Analysis, Text Mining

Identifying univariate outlier detection using k-means/DBSCAN where points outside one big cluster are regarded as outliers but they tend to capture spikes in series more than shifts
References:
https://towardsdatascience.com/density-based-algorithm-for-outlier-detection-8f278d2f7983
https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012035/pdf
https://towardsdatascience.com/time-series-of-price-anomaly-detection-13586cd5ff46
Things that can potentially capture level shift are (raw tested today):
1.	Isotonic regression: its a monotonic regression fitting a free form line. By specifying 'increasing=False' among parameters: https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html we can fit level shifting line without making it strictly monotonic.
2.	Rolling std deviation/variance: http://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/; https://stackoverflow.com/questions/45487229/identifying-an-event-based-on-a-jump-in-rolling-variance-in-python
3.	Decision tree regression: https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
4.	Identified method as:
Step1 - Fit a Decision Tree Regressor to figure out steps/levels in the data.
Step2 - Calculate rolling standard deviation in steps/levels to point out the jumps/dips.
Step3 - Capture peaks in standard deviations to identify change of level dates.
5.	One example, for motherboard counts, is shown as:
(Legend - Original count: dark blue lines, steps/levels: light blue dots, rolling std dev: green dots)

More References:
https://arxiv.org/pdf/1301.3781.pdf
https://arxiv.org/pdf/1310.4546.pdf
https://radimrehurek.com/gensim/models/word2vec.html
