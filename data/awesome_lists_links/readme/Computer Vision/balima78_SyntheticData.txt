# SyntheticData
a draft for a CRAN task view on Synthetic Data

**Maintainer:** Bruno A Lima                                  
**Contact:**    balima78 at gmail.com                      
**Version:**    2021-05-09                                     
**URL:**        *no link to CRAN yet*   

<div>

Testing and validating applications or data products requires the use of data that is not always available to us. 
As an alternative to real data, we have the possibility to generate fake or synthetic data in a format similar to real data. This task view collects information on R packages with functionalities to generate synthetic data.

[Wikipedia](https://en.wikipedia.org/wiki/Synthetic_data) defines synthetic data as follows: Synthetic data is "any production data applicable to a given situation that are not obtained by direct measurement" according to the McGraw-Hill Dictionary of Scientific and Technical Terms; where Craig S. Mullins, an expert in data management, defines production data as "information that is persistently stored and used by professionals to conduct business processes."

Base **R** allow us to genetare data vectors according to different distributions 
with functions as: `rnorm`, `rexp`, `rpois`, `runif`, `rmultinom`, `sample`... This CRAN task view contains a list of packages that can be used to generate synthetic data. Also, several packages to deal with unbalanced data are listed.

The [rOpenSci Task View: Open Data](https://github.com/ropensci-archive/opendata#cran-task-view-open-data) repository provides information about using R to obtain, parse, manipulate, create, and share open data. Moreover the [WebTechnologies Task View](https://cran.r-project.org/web/views/WebTechnologies.html) addresses how to obtain and parse web-based data. Furthermore, a curated list of data packages can be found at [ropensci-archive/data-packages](https://github.com/ropensci-archive/data-packages). 

The development of this task view is fairly new and still in its early stages and therefore subject to changes. Suggestions for additions and extensions to the task view maintainer are welcomed.

**Builders**

- [charlatan](https://github.com/ropensci/charlatan) - Makes fake data; inspired from and borrowing some code from Python's faker
- [conjurer](https://www.foyi.co.nz/posts/documentation/documentationconjurer/) -Builds synthetic data applicable across multiple domains. This package also provides flexibility to control data distribution to make it relevant to many industry examples as described in the [vignette](https://cran.r-project.org/web/packages/conjurer/vignettes/introduction_to_conjurer.html).
- [datasynthR](https://github.com/jknowles/datasynthR) (not on CRAN) - Functions to procedurally generate synthetic data in R for testing and collaboration. Allows the user to generate data of known distributional properties with known correlation structures. This is useful for testing statistical model data, building functions to operate on very large datasets, or training others in using R!
- [fabricatr](https://github.com/DeclareDesign/fabricatr) - This package helps researchers imagine what data will look like before they collect it. Researchers can evaluate alternative analysis strategies, find the best one given how the data will look, and precommit before looking at the realized data.
- [fakeR](https://cran.r-project.org/web/packages/fakeR/vignettes/my-vignette.html) - Simulates Data from a Data Frame of Different Variable Types. The package contains the functions `simulate_dataset` and `simulate_dataset_ts` to simulate time-independent and time-dependent data. It randomly samples character and factor variables from contingency tables and numeric and ordered factors from a multivariate normal distribution. It currently supports the simulation of stationary and zero-inflated count time series.
- [gendata](https://cran.r-project.org/web/packages/gendata/gendata.pdf) - Generate and Modify Synthetic Datasets. Set of functions to create datasets using a correlation matrix.
- [humanleague](https://github.com/virgesmith/humanleague): Synthetic Population Generator. An R package for microsynthesising populations from marginal and (optionally) seed data. 
- [OpenSDPsynthR](https://github.com/opensdp/OpenSDPsynthR) (not on CRAN) - Generate synthetic education data that is realistic for use by analysts across the education sector. Synthetic data should be able to be generated on-demand and responsive to inputs from the user.
- [rstyles](https://cran.r-project.org/web/packages/rstyles/index.html) - Package allows to generate simulated datasets using algorithms that mimic different response styles to survey questions.
- [sdglinkage](https://rdrr.io/cran/sdglinkage/) - Synthetic Data Generation for Linkage Methods Development. A tool for synthetic data generation that can be used for linkage method development, with elements of i) gold standard file with complete and accurate information and ii) linkage files that are corrupted as we often see in raw dataset.
- [SimMultiCorrData](https://github.com/AFialkowski/SimMultiCorrData) - The goal of SimMultiCorrData is to generate continuous (normal or non-normal), binary, ordinal, and count (Poisson or Negative Binomial) variables with a specified correlation matrix. It can also produce a single continuous variable. This package can be used to simulate data sets that mimic real-world situations (i.e. clinical data sets, plasmodes, as in Vaughan et al., 2009).
- [simPop](https://www.jstatsoft.org/article/view/v079i10) - Simulation of Complex Synthetic Data Information. Tools and methods to simulate populations for surveys based on auxiliary data. The tools include model-based methods, calibration and combinatorial optimization algorithms.
- [simstudy](https://cran.r-project.org/web/packages/simstudy/vignettes/simstudy.html) - This package has a collection of functions that allow users to generate simulated data sets in order to explore modeling techniques or better understand data generating processes.
- [synthpop](https://www.synthpop.org.uk/index.html) - This package for R allows users to create synthetic versions of confidential individual-level data for use by researchers interested in making inferences about the population that the data represent. It allows the synthesis process to be customised in many different ways according to the characteristics of the data being synthesised.
- [MicSim](https://microsimulation.pub/articles/00105) - Performing Continuous-Time Microsimulation. This entry-level toolkit allows performing continuous-time microsimulation for a wide range of demographic applications. Given a initial population, mortality rates, divorce rates, marriage rates, education changes, etc. and their transition matrix can be defined and included for the simulation of future states of the population.
- [sms](https://www.jstatsoft.org/article/view/v068i02) - Spatial Microsimulation. Produce small area population estimates by fitting census data to survey data.
- [saeSim](https://wahani.github.io/saeSim/) - Tools for the simulation of data in the context of small area estimation. Combine all steps of your simulation - from data generation over drawing samples to model fitting - in one object.

**Specific types of data**

- [survsim](https://www.jstatsoft.org/article/view/v059i02) - Simulation of Simple and Complex Survival Data. Simulation of simple and complex survival data including recurrent and multiple events and competing risks.
- [sim.survdata()](https://cran.r-project.org/web/packages/coxed/vignettes/simulating_survival_data.html) function allows to generate a survival dataset. This function belongs to the [coxed](https://github.com/jkropko/coxed) package.
- [fakeR](https://cran.r-project.org/web/packages/fakeR/vignettes/my-vignette.html) - Simulates Data from a Data Frame of Different Variable Types. The package contains the functions `simulate_dataset` and `simulate_dataset_ts` to simulate time-independent and time-dependent data. It randomly samples character and factor variables from contingency tables and numeric and ordered factors from a multivariate normal distribution. It currently supports the simulation of stationary and zero-inflated count time series.
- [synthesis](https://github.com/zejiang-unsw/synthesis#readme) - Synthetic data generator. Generate synthetic time series from commonly used statistical models, including linear, nonlinear and chaotic systems.

**Imbalanced data**

- [ebal](https://web.stanford.edu/~jhain/ebalancepage.html) - Entropy reweighting to create balanced samples. Implements entropy balancing, a data preprocessing procedure that allows users to reweight a dataset such that the covariate distributions in the reweighted data satisfy a set of user specified moment conditions. This can be useful to create balanced samples in observational studies with a binary treatment where the control group data can be reweighted to match the covariate moments in the treatment group. Entropy balancing can also be used to reweight a survey sample to known characteristics from a target population.
- [imbalance](https://github.com/ncordon/imbalance) - Preprocessing Algorithms for Imbalanced DatasetsClass imbalance usually damages the performance of classifiers. This package provides a set of tools to work with imbalanced datasets: novel oversampling algorithms, filtering of instances and evaluation of synthetic instances.
- [IRIC](https://github.com/shuzhiquan/IRIC) (not on CRAN) - An R library for binary imbalanced classification. Integrates a wide set of solutions for imbalanced binary classification.
- [ROSE](https://journal.r-project.org/archive/2014/RJ-2014-008/index.html) - Random Over-Sampling Examples. Thi package provides functions to deal with binary classification problems in the presence of imbalanced classes. Synthetic balanced samples are generated according to ROSE (Menardi and Torelli, 2013). Functions that implement more traditional remedies to the class imbalance are also provided, as well as different metrics to evaluate a learner accuracy. These are estimated by holdout, bootstrap or cross-validation methods.
- [smotefamily](https://cran.r-project.org/web/packages/smotefamily/smotefamily.pdf) - A Collection of Oversampling Techniques for Class Imbalance Problem Based on SMOTE. A collection of various oversampling techniques developed from SMOTE is provided. SMOTE is a oversampling technique which synthesizes a new minority instance between a pair of one minority instance and one of its K nearest neighbor.
- [themis](https://github.com/tidymodels/themis) - Extra Recipes Steps for Dealing with Unbalanced Data. A dataset with an uneven number of cases in each class is said to be unbalanced. Many models produce a subpar performance on unbalanced datasets. A dataset can be balanced by increasing the number of minority cases using [SMOTE 2011](https://arxiv.org/abs/1106.1813), [Borderline-SMOTE 2005](https://link.springer.com/chapter/10.1007/11538059_91); and [ADASYN 2008](https://ieeexplore.ieee.org/document/4633969); or by decreasing the number of majority cases using [NearMiss 2003](https://www.site.uottawa.ca/~nat/Workshop2003/jzhang.pdf); or [Tomek link removal 1976](https://ieeexplore.ieee.org/document/4309452);.
- [unbalanced](https://github.com/dalpozz/unbalanced) - Racing for Unbalanced Methods Selection. This R package implements some well-known techniques for unbalanced classification tasks and provides a racing strategy to adaptively select the best methods for a given dataset, classification algorithms and accuracy measure adopted.

**Miscellaneous**

- [bindata](http://finzi.psych.upenn.edu/library/bindata/html/00Index.html) - Generation of correlated artificial binary data.
- [GenOrd](https://rdrr.io/cran/GenOrd/#vignettes) - Simulation of Discrete Random Variables with Given Correlation Matrix and Marginal Distributions. A gaussian copula based procedure for generating samples from discrete random variables with prescribed correlation matrix and marginal distributions.
- [fakir](https://thinkr-open.github.io/fakir/) (not on CRAN) - The goal of {fakir} is to provide fake datasets that can be used to teach R.
- [MultiOrd](https://www.tandfonline.com/doi/abs/10.1080/03610918.2013.824097?journalCode=lssp20) - An R Package for Generating Correlated Ordinal Data.
- [NestedCategBayesImpute](https://cran.r-project.org/web/packages/NestedCategBayesImpute/NestedCategBayesImpute.pdf) - Modeling, Imputing and Generating Synthetic Versions of Nested Categorical Data in the Presence of Impossible Combinations. This tool set provides a set of functions to fit the nested Dirichlet process mixture of products of multinomial distributions (NDPMPM) model for nested categorical household data in the presence of impossible combinations. It has direct applications in imputing missing values for and generating synthetic versions of nested household data.
- [NHSRdatasets](https://cran.r-project.org/web/packages/NHSRdatasets/index.html) - Free United Kingdom National Health Service (NHS) and other healthcare, or population health-related data for education and training purposes. This package contains synthetic data based on real healthcare datasets, or cuts of open-licenced official data.
- [PoisBinOrdNonNor](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.5362) - Generation of a chosen number of count, binary, ordinal, and continuous random variables, with specified correlations and marginal properties.
- [psychmeta](https://github.com/psychmeta/psychmeta) - This package provides tools for computing bare-bones and psychometric meta-analyses and for generating psychometric data for use in meta-analysis simulations.
- [sgr](https://cran.r-project.org/web/packages/sgr/sgr.pdf) - Sample Generation by Replacement simulations. The package can be used to perform fake data analysis according to the sample generation by replacement approach. It includes functions for making simple inferences about discrete/ordinal fake data. The package allows to study the implications of fake data for empirical results.
- [synthACS](http://cran.nexr.com/web/packages/synthACS/synthACS.pdf) - Synthetic Microdata and Spatial MicroSimulation Modeling for ACS Data. Provides access to curated American Community Survey (ACS) base tables. Builds synthetic micro-datasets at any user-specified geographic level with ten default attributes; and, conducts spatial microsimulation modeling (SMSM) via simulated annealing.
- [SynthTools](https://github.com/RTIInternational/SynthTools) - Tools and Tests for Experiments with Partially Synthetic Data Sets. A set of functions to support experimentation in the utility of partially synthetic data sets. All functions compare an observed data set to one or a set of partially synthetic data sets derived from the observed data to (1) check that data sets have identical attributes, (2) calculate overall and specific variable perturbation rates, (3) check for potential logical inconsistencies, and (4) calculate confidence intervals and standard errors of desired variables in multiple imputed data sets. Confidence interval and standard error formulas have options for either synthetic data sets or multiple imputed data sets.
- [wakefield](https://github.com/trinker/wakefield) - A package designed to quickly generate random data sets. The user passes `n` (number of rows) and predefined vectors to the `r_data_frame` function to produce a `dplyr::tbl_df` object.


</div>

### CRAN packages:

- [bindata](https://cran.r-project.org/web/packages/bindata/index.html)
- [charlatan](https://cran.r-project.org/web/packages/charlatan/index.html)
- [conjurer](https://cran.r-project.org/web/packages/conjurer/index.html)
- [coxed](https://cran.r-project.org/web/packages/coxed/index.html)
- [datasynthR](https://github.com/jknowles/datasynthR)
- [ebal](https://cran.r-project.org/web/packages/ebal/index.html)
- [fabricatr](https://cran.r-project.org/web/packages/fabricatr/index.html)
- [fakeR](https://cran.r-project.org/web/packages/fakeR/index.html)
- [fakir](https://github.com/ThinkR-open/fakir)
- [gendata](https://cran.r-project.org/web/packages/gendata/index.html)
- [GenOrd](https://cran.r-project.org/web/packages/GenOrd/index.html)
- [humanleague](https://cran.r-project.org/web/packages/humanleague/index.html)
- [imbalance](https://cran.r-project.org/web/packages/imbalance/index.html)
- [IRIC](https://github.com/shuzhiquan/IRIC)
- [MicSim](https://cran.r-project.org/web/packages/MicSim/index.html)
- [MultiOrd](https://cran.r-project.org/web/packages/MultiOrd/index.html)
- [NestedCategBayesImpute](https://cran.r-project.org/web/packages/NestedCategBayesImpute/index.html)
- [NHSRdatasets](https://cran.r-project.org/web/packages/NHSRdatasets/index.html)
- [OpenSDPsynthR](https://github.com/OpenSDP/OpenSDPsynthR)
- [PoisBinOrdNonNor](https://cran.r-project.org/web/packages/PoisBinOrdNonNor/index.html)
- [psychmeta](https://cran.r-project.org/web/packages/psychmeta/index.html)
- [ROSE](https://cran.r-project.org/web/packages/ROSE/index.html)
- [rstyles](https://cran.r-project.org/web/packages/rstyles/index.html)
- [saeSim](https://cran.r-project.org/web/packages/saeSim/index.html)
- [sdglinkage](https://cran.r-project.org/web/packages/sdglinkage/index.html)
- [sgr](https://cran.r-project.org/web/packages/sgr/index.html)
- [SimMultiCorrData](https://cran.r-project.org/web/packages/SimMultiCorrData/index.html)
- [simPop](https://cran.r-project.org/web/packages/simPop/index.html)
- [simstudy](https://cran.r-project.org/web/packages/simstudy/index.html)
- [smotefamily](https://cran.r-project.org/web/packages/smotefamily/index.html)
- [sms](https://cran.r-project.org/web/packages/sms/index.html)
- [survsim](https://cran.r-project.org/web/packages/survsim/index.html)
- [synthACS](https://cran.r-project.org/web/packages/synthACS/index.html)
- [synthesis](https://cran.r-project.org/web/packages/synthesis/index.html)
- [synthpop](https://cran.r-project.org/web/packages/synthpop/index.html)
- [SynthTools](https://cran.r-project.org/web/packages/SynthTools/index.html)
- [themis](https://cran.r-project.org/web/packages/themis/index.html)
- [unbalanced](https://cran.r-project.org/web/packages/unbalanced/index.html)
- [wakefield](https://cran.r-project.org/web/packages/wakefield/index.html)

### Related links:

-   [CRAN Task Views](https://cran.r-project.org/web/views/)
-   [How to write CRAN Task Views](https://cran.r-project.org/web/packages/ctv/vignettes/ctv-howto.pdf)
-   [CRAN Task View: Missing Data](https://cran.r-project.org/web/views/MissingData.html)
-   [CRAN Task View: Official Statistics & Survey Methodology](https://cran.r-project.org/web/views/OfficialStatistics.html)
-   [CRAN Task View: Web Technologies and Services](https://cran.r-project.org/web/views/WebTechnologies.html)
-   [awesome-data-synthesis](https://github.com/joofio/awesome-data-synthesis)
-   [GitHub repository for this TaskView](https://github.com/balima78/SyntheticData)
-   [HEADS-FMUP](https://github.com/HEADS-FMUP)