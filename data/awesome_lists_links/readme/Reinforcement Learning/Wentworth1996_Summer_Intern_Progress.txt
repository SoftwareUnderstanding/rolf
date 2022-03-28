# Yuxiang's Intern at Millennium, Summer 2020, as Quantitative Developer
This project is mainly to record my internship at Millennium, i.e. the papers I read, the tools I learned, the toy examples I experimented on, the ideas I proposed for trading futures & the questions, etc. I won't include any intranet resources here, neither will I fork any internship codes. 

For the following two reasons, I built this project. First, recording my daily progress helps me better reflect and retrospect. Even though I've learned a library, e.g. Dask, a priori, I still need all the documentation and examples for help when I really work on Dask, since I can barely remember all the interfaces by heart. For this reason, I will also include all the links I looked for help. Similarly, recording the main ideas & methods of every paper I read is essential for building my own trading strategy, especially because an internship strategy is almost a "combination" and "implementation". Personally, in the first two years I start working as a quantitative researcher, I will read those classic works of my predecessors: Standing on the shoulders of giants is a good way for research. Secondly, this will help me better demonstrate my work to my mentor, Alex.

The materials I used are basically written in English. However, out of some reasons, I inevitably included a few Chinese and Frech resources.

## Pre-internship learnings
Following Alex's and Jing's instructions, I armed myself familiar with the following topics:

### Dask
[The Dask Documentation](https://docs.dask.org/en/latest/), [the Dask.Distributed Documentation](https://distributed.dask.org/en/latest/) and [the examples](https://github.com/dask/dask-tutorial) are substantial materials for learning Dask. I've learned the APIs for Array, Bag, Dataframe, Delayed. But I haven't learned Distributed in details. It seems an advanced API for scheduling. I read until Build Understanding/Efficiency. See also [Stanford](http://cs149.stanford.edu/winter19/).

### Rolling Continuous Futures
- [Paper I](https://onlinelibrary.wiley.com/doi/abs/10.1002/fut.20373) reviewd in details. The five methods result in the same (statistically) time series. The parametric F test, the non-parametric Kruskal-Wallis test, and Brown Forsythe's statistic (see wiki) are important tests for evaluating the similarity.

### Leveraging time series momentum to trade futures
[Notorious AQR paper](http://docs.lhpedersen.com/TimeSeriesMomentum.pdf). Typical old Chicago school paper with high school maths. Formula in Page 17 claimed that the return is proportional to the position. Not very clear: what is "cross section", ex-ante vol, how to distinguish speculators and hedgers. Reviewed up to page 28. 

### Reinforcement Learning and Deep Reinforcement Learning
[The Gomoko project](https://github.com/junxiaosong/AlphaZero_Gomoku) basically implements the AlphaZero to Gomoko Games. It provides a RL coding framework.  [Atari](https://arxiv.org/abs/1312.5602) the first DQN. See also [cutting plane method](https://www.math.cuhk.edu.hk/course_builder/1415/math3220/L5.pdf), [simplex table](https://personal.utdallas.edu/~scniu/OPRE-6201/documents/LP06-Simplex-Tableau.pdf).

### DRL for trading
Many papers... some been read in details but not anticipate to implement. [Paper I] (https://arxiv.org/abs/2004.06627) it is rather a graduation thesis but the points are clearly presented.
### LSTM model 
[Page](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) reviewd. Better find some more technical papers. LSTM, or more generally, they recurrent network, seems the only DL model for time series.

### SQL
[Page](https://www.liaoxuefeng.com/wiki/1016959663602400/1017803857459008) contains a tutorial for SQLAlchemy. Easy! [Berkeley SQL](https://www.stat.berkeley.edu/~spector/sql.pdf) contains the fundamental query sentences. 

### Airflow
[Airflow](https://airflow.apache.org/) page contains the official documentation. I haven't gone into details since it is too practical. Better consult this page when actually working on it. 

### Python OOP Review
[Guide](https://www.liaoxuefeng.com/wiki/1016959663602400) is a Python tutorial written in Chinese. Function, OOP, SQL parts reviewed.

### Miscellanea
- [Traditional trading strategies overview](https://www.quantconnect.com/tutorials/strategy-library/term-structure-effect-in-commodities). [Traditional methods to combine alphas1](https://zhuanlan.zhihu.com/p/38340204), [alphas2](https://zhuanlan.zhihu.com/p/38340466).

- What is the "cross-section"? [This link](https://stats.stackexchange.com/questions/40852/what-is-cross-section-in-cross-section-of-stock-return/40857) explained the concept in a understandable way but that's not enough. If the study of cross-section aims to answer the question why stock A earns higher/lower returns than stock B, then does the cross-section simply rank the returns in a universe?

- (June 11) Dow plunges nearly 7% as stocks post worst session since March. I really can't figure out why, 
although many plunges in history are equally difficult to explain. The resurgent coronavirus cases due to past protests are not sufficient to explain since the negative sentiment that appeared many days ago should have been slowly digested day by day. 
I would rather attribute this to the behavior of certain funds. But could a few number of funds be that powerful?

- [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance), a mathematically-rigorous method to measure the similarities between two (continuous) time series, i.e. curve. The scores maybe useful to construct an arbitrage strategy, e.g. pair trading. [Timing](https://www.sciencedirect.com/science/article/pii/S0005109813003609) okay...academic style pair-trading timing rule.

- Arbitrage. [Paper I](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1404905) (cointegration from Multivariate Ornstein-Uhlenbeck process's perspectives) reviewd in details. [Vectorization](https://en.wikipedia.org/wiki/Vectorization_(mathematics)) provides an introduction to the related algebra. [Complex eigenvector transform](http://www.sci.wsu.edu/math/faculty/schumaker/Math512/512F10Ch2B.pdf) explains the naive matrix pseudo-diagonalization in this paper.
[Values of commodity futures](https://www.tandfonline.com/doi/pdf/10.2469/faj.v62.n2.4084?needAccess=true) interesting abstract but haven't read.

- [Fuzzy systems](https://arxiv.org/abs/1401.1888) (and Part II, III). Very interesting papers. The author claimed that he earned 100M in the market using this model, but out of pure academic pursuits he then became a professor and made this research public. Beautiful theories logically explained, fabulous experimental results.

## Behavioral trainings: Etiquette and Norms
TBC on June 15.
