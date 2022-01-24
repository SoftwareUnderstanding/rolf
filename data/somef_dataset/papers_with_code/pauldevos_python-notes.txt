# Python Resources

1. Installing and Managing Python Versions
   - [pyenv](https://github.com/pyenv/pyenv) - ([tutorial](https://akrabat.com/creating-virtual-environments-with-pyenv/)), ([Deep dive into how pyenv actually works by leveraging the shim design pattern](https://mungingdata.com/python/how-pyenv-works-shims/))
2. Managing virtual environments
   - [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)
   - [venv](https://docs.python.org/3/library/venv.html) - ([tutorial](https://www.youtube.com/watch?v=Kg1Yvry_Ydk))
   - conda
3. Code Editor

   - I recommend using [Visual Studio Code](https://code.visualstudio.com/download) and here's a few tutorials on how to set up VS Code for Python projects.
   - To get set up with Visual Studio Code for Python projects I recommend these tutorials:
     - [YouTube: Cory Schafer - Visual Studio Code (Mac) - Setting up a Python Development Environment and Complete Overview](https://www.youtube.com/watch?v=06I63_p-2A4)
     - [YouTube: Traversy Media - Setting Up VSCode For Python Programming](https://www.youtube.com/watch?v=W--_EOzdTHk)
     - [Miguel Grinberg Visual Studio Code Setup](https://www.youtube.com/watch?v=UXqiVe6h3lA)
   - VS Code Pluin Recommendations for better development experience.
   - EditorConfig

     - Generate .editorconfig in a directory in your project.

   - [YouTube: Traversy Media - 15 VS Code Extensions For Front-End Developers in 2019](https://www.youtube.com/watch?v=LdF2RcelRg0)

   - For Data Science projects, I also recommend Jupyter Notebooks as it's the defacto quick prototyping "IDE" for Data Science, Data Viz, and other similar projects that need non-technical stakeholder input. There definitely are ways to do "development" in Notebooks right as well. I'll try to share some resources on that. You will often use Notebooks in conjuction with a text editor like VS Code. There is JupyterLab which does attempt to combine notebooks and a text editor into one "environment" but as of March 2020 I haven't really been impressed with it to abandon using VS Code or Atom or Sublime for what JupyterLab can offer. To me the essential tool is a great text editor slash IDE and VS Code is that.

4. Git
   - [Git Tutorial for Beginners: Command-Line Fundamentals](https://www.youtube.com/watch?v=HVsySz-h9r4)
   - [Git Tutorial: Fixing Common Mistakes and Undoing Bad Commits](https://www.youtube.com/watch?v=FdZecVxzJbk)
   - [YouTube: Git Playlist Corey Schafer](https://www.youtube.com/watch?v=HVsySz-h9r4&list=PL-osiE80TeTuRUfjRe54Eea17-YfnOOAx)
5. Testing

   - There are many Python packages out there you could use. The ones I like are:
     - pytest
     - coverage
     - pytest-cov
   - [YouTube: Overview of Pytest, Unittest, Coverage, and Pytest-cov](https://www.youtube.com/watch?v=7BJ_BKeeJyM)
   - Pytest
     - [YouTube: Python Testing 101 with pytest](https://www.youtube.com/watch?v=etosV2IWBF0)
     - [YouTube: Python Testing 201 with pytest](https://www.youtube.com/watch?v=fv259R38gqc)

6. Deploying Apps

   - Synchronous Framework: Flask
   - Asynchronous Framework: FastAPI
   - AWS ML: SageMaker
   - AWS Serverless: Zappa
   - AWS web framework: Chalice

7. Utilities and Automation Scripts

   - Automation Scripts
   - Subprocess
   - Click
   - Path and Utilities

     - Pathlib
     - os
     - sys
     - devtools

8. Optional:
   - Type Hints
     - [Typer](https://github.com/tiangolo/typer)
   - CLI Libraries

## Intermediate Level Overviews of Python

- [Python Tips Docs](http://book.pythontips.com/en/latest/)
- [Python Anti-Patterns](https://docs.quantifiedcode.com/python-anti-patterns/)
- [Berkeley Python Bootcamp](https://www.youtube.com/watch?v=P5BHTrluu1M&list=PLKW2Azk23ZtSeBcvJi0JnL7PapedOvwz9&index=1)
- [Effective Python](https://www.amazon.com/Effective-Python-Specific-Software-Development/dp/0134034287) - [code](https://github.com/bslatkin/effectivepython)
- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)
- [HackerRank](https://www.hackerrank.com/)
- [Python 3 official docs](https://docs.python.org/3/)
- [Data-Flair-Python-Training](https://data-flair.training/blogs/python-tutorials-home/)
- [List of Python APIs with Examples](https://www.programcreek.com/python/index/module/list)

## Writing Python Modules - Some Essential Python Libraries

- [The Hitchhiker's Guide to Packaging](https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/index.html)
- [The Hitchhiker's Guide to Structuring Your Project](https://docs.python-guide.org/writing/structure/)
- [setup.py](https://github.com/kennethreitz/setup.py)
- [How To Package Your Python Code](https://python-packaging.readthedocs.io/en/latest/)
- [Python Application Layouts: A Reference](https://realpython.com/python-application-layouts/)

## OOP

- [Python vs Java Classes](https://realpython.com/oop-in-python-vs-java/)
- [A Byte of Python](https://python.swaroopch.com/oop.html)
- [The Python Tutorial - Classes](https://docs.python.org/3/tutorial/classes.html)
- [Object-Oriented Programming (OOP) in Python 3](https://realpython.com/python3-object-oriented-programming/)
- [Tutorials Point - Objected Oriented Python](https://www.tutorialspoint.com/python/python_classes_objects.htm)
- [Python-Textbook](https://python-textbok.readthedocs.io/en/1.0/Object_Oriented_Programming.html)
- [Python by Programiz - OOP](https://www.programiz.com/python-programming/object-oriented-programming)
- [Object Oriented Design - Niko Wilbert](https://python.g-node.org/python-summerschool-2013/_media/wiki/oop/oo_design_2013.pdf)
- [Python 101 - Object Oriented Programming - Part 1](https://medium.com/the-renaissance-developer/python-101-object-oriented-programming-part-1-7d5d06833f26)
- [Improve Your Python: Python Classes and Object Oriented Programming](https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/)
- [Raymond Hettinger - Python's Class Development Toolkit](https://www.youtube.com/watch?v=HTLu2DFOdTg)
- [OOP - Trying To Design A Good Class Structure)](https://stackoverflow.com/questions/39922553/oop-trying-to-design-a-good-class-structure)

## Building RESTful APIs

- [Creating Web APIs with Python and Flask](https://programminghistorian.org/en/lessons/creating-apis-with-python-and-flask)
- [Using Elasticsearch with Python and Flask](https://dev.to/aligoren/using-elasticsearch-with-python-and-flask-2i0e)

## Design Patterns

- [A Collection of Design Patterns in Python](https://github.com/faif/python-patterns)
- [Toptal - Python Design Patterns](https://www.toptal.com/python/python-design-patterns)
- [Python Patterns](https://github.com/faif/python-patterns) - A collection of design patterns/idioms in Python
- [10 Common Software Architectural Patterns in a nutshell](https://towardsdatascience.com/10-common-software-architectural-patterns-in-a-nutshell-a0b47a1e9013)

## Building RESTful API Wrappers

- [Designing a RESTful API with Python and Flask - Miguel Grinberg](https://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask)
- [How To Use Web APIs in Python 3](https://www.digitalocean.com/community/tutorials/how-to-use-web-apis-in-python-3)
- [Building and Documenting Python REST APIs With Flask and Connexion](https://realpython.com/flask-connexion-rest-api/)
- [API Integration in Python](https://realpython.com/api-integration-in-python/)
- [How to Use Restful Web APIs in Python](https://code.tutsplus.com/articles/how-to-use-restful-web-apis-in-python--cms-29493)

## Algorithms in Python

- [The Algorithms](https://github.com/TheAlgorithms/Python)

## Testing in Python

- [Jes Ford - Getting Started Testing in Data Science - PyCon 2019](https://www.youtube.com/watch?v=0ysyWk-ox-8)
- [Eric J Ma Best Testing Practices for Data Science PyCon 2017](https://www.youtube.com/watch?v=yACtdj1_IxE) - [Github](https://github.com/ericmjl/data-testing-tutorial)

## Virtual Environments

Creating Your First Project & Virtual Environment

```
# Create a directory and enter into it
mkdir myproject && cd myproject

# create a python env
pyvenv venv

# Put the venv in your .gitignore:
git init
echo 'venv' > .gitignore
```

Doing this keeps your virtual environment out of source control (Git).

```
# Activate the environment:
source venv/bin/activate

# Install packages
pip install requests bs4

# Freeze the requirements:
pip freeze > requirements.txt

# Check requirements.txt into source control:
git add requirements.txt
```

# favorite-python-resources

A list of some favorite libraries in Python as well as specific areas of Python e.g. Data Science, Machine Learning, etc. that I've found helpful.

Popular Python APIs (with code examples)

- [Module List](https://www.programcreek.com/python/index/module/list)

Popular Python ML APIs

- [Skymind - Python APIs](https://skymind.ai/wiki/python-ai)

### Python Videos/Tutorials

- [Raymond Hettinger - Beyond PEP 8 -- Best practices for beautiful intelligible code - PyCon 2015](https://www.youtube.com/watch?v=wf-BqAjZb8M)
- [CS212 - Design of Computer Programs by Peter Norvig](https://classroom.udacity.com/courses/cs212)
- [Cory Shafer Python OOP - 6 Video Series](https://www.youtube.com/watch?v=ZDa-Z5JzLYM&list=PL-osiE80TeTsqhIuOqKhwlXsIBIdSeYtc)

### Python Libraries

---

#### Data Science Workflows

- [Data Science Workflows using Docker Containers(YouTube)](https://www.youtube.com/watch?v=oO8n3y23b6M)
- [How Docker Can Make You A More Effective Data Scientist](https://towardsdatascience.com/how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5)

#### Linear Algebra

- [fast.ai - Linear Algebra](https://github.com/fastai/numerical-linear-algebra)
- [numerical-linear-algebra](https://github.com/fastai/numerical-linear-algebra)

#### Python Data Science Books

- [Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow](https://github.com/ageron/handson-ml)
- [Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow 2 (2nd Ed)](https://github.com/ageron/handson-ml2)
- [Python Machine Learning (2nd edition) - Sebastian Raschka](https://github.com/rasbt/python-machine-learning-book-2nd-edition)
- [Introduction to Artificial Neural Networks and Deep Learning - Sebastian Raschka](https://github.com/rasbt/deep-learning-book) [code](https://github.com/rasbt/deeplearning-models)
- [Pattern Classification - Sebastian Raschka](https://github.com/rasbt/pattern_classification)
- [Agile Data Science 2.0](https://github.com/rjurney/Agile_Data_Code_2)
- [Deep Learning with Python - François Chollet](https://github.com/fchollet/deep-learning-with-python-notebooks)
- [PythonDataScienceHandbook](https://github.com/jakevdp/PythonDataScienceHandbook)
- [practical-machine-learning-with-python](https://github.com/dipanjanS/practical-machine-learning-with-python)
- [ISLR](http://www-bcf.usc.edu/~gareth/ISL/) - [ISLR in Python](https://github.com/JWarmenhoven/ISLR-python) - An Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani, 2013): Python code
- [Data Science from Scratch](https://github.com/joelgrus/data-science-from-scratch)
- [Data-Analysis-and-Machine-Learning-Projects](https://github.com/rhiever/Data-Analysis-and-Machine-Learning-Projects)
- [Deep Learning Tutorials](https://github.com/lisa-lab/DeepLearningTutorials) - Repository of teaching materials, code, and data for my data analysis and machine learning projects.
- [TensorFlow Tutorials](https://github.com/pkmital/tensorflow_tutorials)

---

#### My Favorite Tools

- [Yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) - Visual analysis and diagnostic tools to facilitate machine learning model selection. http://www.scikit-yb.org/
- [scikit-plot](https://github.com/reiinakano/scikit-plot) - An intuitive library to add plotting functionality to scikit-learn objects.
- [missingno](https://github.com/ResidentMario/missingno) - Visualize missing data
- [Holoviews](https://github.com/ioam/holoviews) - With Holoviews, your data visualizes itself
- [altair](https://altair-viz.github.io/index.html) - ([Github](https://github.com/altair-viz/altair)) - Declarative statistical visualization library for Python
- [Pandas Styling](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html) - [Mode Blog](https://mode.com/example-gallery/python_dataframe_styling/)
- [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling) - Create HTML profiling reports from pandas DataFrame objects

### Ones to Check out

- [mlens](https://github.com/flennerhag/mlens) - ML-Ensemble – high performance ensemble learning http://ml-ensemble.com
- [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) - Sequential model-based optimization with a `scipy.optimize` interface https://scikit-optimize.github.io

---

#### Machine Learning Libraries

- [Lime](https://github.com/marcotcr/lime) - Lime: Explaining the predictions of any machine learning classifier
- [shap](https://github.com/slundberg/shap) - A unified approach to explain the output of any machine learning model.
- [eli5](https://github.com/TeamHG-Memex/eli5) - A library for debugging/inspecting machine learning classifiers and explaining their predictions
- [catboost](https://github.com/catboost/catboost) - https://github.com/catboost/catboost
- [interpretable_machine_learning_with_python](https://github.com/jphall663/interpretable_machine_learning_with_python) - Practical techniques for training interpretable ML models, explaining ML models, and debugging ML models.
- [awesome-machine-learning-interpretability](https://github.com/jphall663/awesome-machine-learning-interpretability) - A curated list of awesome machine learning interpretability resources.
- [gym](https://github.com/openai/gym) - A toolkit for developing and comparing reinforcement learning algorithms.
- [Surprise](https://github.com/NicolasHug/Surprise) - A Python scikit for building and analyzing recommender systems
- [kubeflow](https://github.com/kubeflow/kubeflow) - Machine Learning Toolkit for Kubernetes
- [Metrics](https://github.com/benhamner/Metrics)- Machine learning evaluation metrics, implemented in Python, R, Haskell, and MATLAB / Octave
- [MLtest](https://github.com/Thenerdstation/mltest) - Testing framework to simplify writing ML unit tests.
- [scikit-plot](https://github.com/reiinakano/scikit-plot) - An intuitive library to add plotting functionality to scikit-learn objects.
- [featuretools](https://github.com/Featuretools/featuretools) - An open source python framework for automated feature engineering
- [Hands-on Machine Learning Model Interpretation](https://towardsdatascience.com/explainable-artificial-intelligence-part-3-hands-on-machine-learning-model-interpretation-e8ebe5afc608)
- [Machine Learning Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/index.html)
- [Learning Math for Machine Learning](https://blog.ycombinator.com/learning-math-for-machine-learning)
- [Google's Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml/)
- [100 Days of ML Code](https://github.com/Avik-Jain/100-Days-Of-ML-Code)
- [homemade-machine-learning](https://github.com/trekhleb/homemade-machine-learning/blob/master/README.md)
- [category_encoders](https://github.com/scikit-learn-contrib/categorical-encoding) - A library of sklearn compatible categorical variable encoders
- [tensorwatch](https://github.com/microsoft/tensorwatch)
- [Snorkel](https://github.com/snorkel-team/snorkel) - A system for quickly generating training data with weak supervision https://snorkel.org

#### PyTorch

- [PyTorch Examples](https://github.com/pytorch/examples)
- [skorch](https://github.com/skorch-dev/skorch) - A scikit-learn compatible neural network library that wraps pytorch
- [Skorch at Scipy 2019](https://pyvideo.org/scipy-2019/skorch-a-union-of-scikit-learn-and-pytorch-scipy-2019-thomas-fan.html)

#### Hyperparameter Tuners

- [hyperband](https://github.com/zygmuntz/hyperband)
- [hyperas](https://github.com/maxpumperla/hyperas)
- [hyperopt](https://github.com/hyperopt/hyperopt) - Distributed Asynchronous Hyperparameter Optimization in Python http://hyperopt.github.io/hyperopt

---

#### Notebook Tools

- [papermill](https://github.com/nteract/papermill) - Parameterize, execute, and analyze notebooks
- [Notebook Kernels](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels)
- [ipywebrtc](https://github.com/maartenbreddels/ipywebrtc) - WebRTC for Jupyter notebook/lab
- [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20List.html) ([Notebook Example](https://github.com/Chekos/blog-posts/tree/master/altair%20%2B%20ipywidgets))- Interactive Widgets for the Jupyter Notebook
- [itk-jupyter-widgets](https://github.com/InsightSoftwareConsortium/itk-jupyter-widgets) - Interactive Jupyter widgets to visualize images in 2D and 3D.
- [jupytext](https://github.com/mwouts/jupytext) - Jupyter notebooks as Markdown documents, Julia, Python or R scripts
- [voila](https://github.com/QuantStack/voila) - From Jupyter notebooks to standalone web applications and dashboards - [And voilà!](https://blog.jupyter.org/and-voil%C3%A0-f6a2c08a4a93)
- [nb2xls](https://github.com/ideonate/nb2xls) - Convert Jupyter notebook to Excel spreadsheet
- [Jupyter Notebook Extensions](https://towardsdatascience.com/jupyter-notebook-extensions-517fa69d2231)
- [StackOverflow - R & Python in One Jupyter Notebook %%R](https://stackoverflow.com/questions/39008069/r-and-python-in-one-jupyter-notebook)
- [RISE](https://github.com/damianavila/RISE) - RISE allows you to instantly turn your Jupyter Notebooks into a slideshow. No out-of-band conversion is needed, switch from jupyter notebook to a live reveal.js-based slideshow in a single keystroke, and back.
- [Tutorial: Advanced Jupyter Notebooks](https://www.dataquest.io/blog/advanced-jupyter-notebooks-tutorial/)
- [Supporting reproducibility in Jupyter through dataflow notebooks](https://www.youtube.com/watch?v=xUZGP2dGRKQ)
- [Explorations in reproducible analysis with Nodebook](https://www.youtube.com/watch?v=CgJrQXOYIk8)
- [SoS: A polyglot notebook and workflow system for both interactive multilanguage data analysis and batch data processing](https://www.youtube.com/watch?v=U75eKosFbp8)
- [nodebook](https://github.com/stitchfix/nodebook)
- [jupyter-book](https://github.com/jupyter/jupyter-book) - Create an online book with Jupyter Notebooks and Jekyll http://jupyter.org/jupyter-book

### Jupyter Notebook Hubs and Extentions

- [JupyterLab](https://github.com/jupyterlab/jupyterlab) - JupyterLab computational environment
- [jupyterhub](https://github.com/jupyterhub/jupyterhub)
- [Jupyter Notebook Extensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions) - [blog](https://towardsdatascience.com/jupyter-notebook-extensions-517fa69d2231) - [How to write a Jupyter Notebook Extension](https://towardsdatascience.com/how-to-write-a-jupyter-notebook-extension-a63f9578a38c)
- [beakerx](https://github.com/twosigma/beakerx) - Beaker Extensions for Jupyter Notebook
- [nbviewer](https://github.com/jupyter/nbviewer) - nbconvert as a web service: Render Jupyter Notebooks as static web pages

---

#### AutoML

- [MLBox](https://github.com/AxeldeRomblay/MLBox) - MLBox is a powerful Automated Machine Learning python library. https://mlbox.readthedocs.io/en/latest/
- [tpot](https://github.com/EpistasisLab/tpot) - A Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming. http://epistasislab.github.io/tpot/
- [auto-sklearn](https://github.com/automl/auto-sklearn) - Automated Machine Learning with scikit-learn https://automl.github.io/auto-sklearn
- [H20ai](https://github.com/h2oai/h2o-tutorials) - Tutorials and training material for the H2O Machine Learning Platform
- [adanet](https://github.com/tensorflow/adanet) - Fast and flexible AutoML with learning guarantees. https://adanet.readthedocs.io

#### Statistics

- [pgmpy](https://github.com/pgmpy/pgmpy) - [Notebook tutorials](https://github.com/pgmpy/pgmpy/tree/dev/examples) - Python Library for Probabilistic Graphical Models
- [Intro2Stats (Tutorial)](https://github.com/rouseguy/intro2stats/tree/master/notebooks)
- [Thinks Stats 2nd ed](https://github.com/AllenDowney/ThinkStats2)

#### Bayesian Statistics

- [Bayesian-Modelling-in-Python](https://github.com/markdregan/Bayesian-Modelling-in-Python)
- [PyMC3](https://github.com/pymc-devs/pymc3) - Probabilistic Programming in Python: Bayesian Modeling and Probabilistic Machine Learning
- [PyMC3 Resources/Tutorials](https://github.com/pymc-devs/resources) - PyMC3 educational resources - ([Textbook](https://xcelab.net/rm/statistical-rethinking/))
- [arviz](https://github.com/arviz-devs/arviz) - Exploratory analysis of Bayesian models with Python [https://arviz-devs.github.io/arviz/](https://arviz-devs.github.io/arviz/)
- [Bayesian Podcast](https://learnbayesstats.anvil.app/)
- [discourse.pymc.io](https://discourse.pymc.io/)

- [Think Bayes](https://github.com/AllenDowney/ThinkBayes2)

---

#### Data Visualization -- Inspiration and Answers

- [Python Graph Gallery](https://python-graph-gallery.com/)
- [Top 50 matplotlib Visualizations – The Master Plots](https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/)
- [From Data to Viz - Graphing Decisions - code in R](https://www.data-to-viz.com/)
- [The Art of Effective Visualization of Multi-dimensional Data](https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57)
- [Policy Viz - DataViz Books](https://policyviz.com/better-presentations/data-viz-resources/data-viz-books/)
- [Python Plotting for Exploratory Data Analysis](http://pythonplot.com/)
- [Observable - D3 and Other Data Vizualizations](https://beta.observablehq.com/)
- [Our World in Data (Visualizations)](https://ourworldindata.org/)
- [Open Data Science - Data Visualization – How to Pick the Right Chart Type?](https://opendatascience.com/data-visualization-how-to-pick-the-right-chart-type/)
- [Data Viz Project | Collection of data visualizations to get inspired and finding the right type.](https://datavizproject.com/)

#### Data Visualization

- [Yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) - Visual analysis and diagnostic tools to facilitate machine learning model selection. http://www.scikit-yb.org/
- [scikit-plot](https://github.com/reiinakano/scikit-plot) - An intuitive library to add plotting functionality to scikit-learn objects.
- [ipyvolume](https://github.com/maartenbreddels/ipyvolume) - 3d plotting for Python in the Jupyter notebook based on IPython widgets using WebGL
- [Dash](https://dash.plot.ly/) - Written on top of Flask, Plotly.js, and React.js, Dash can be used for highly custom user interfaces in Python
- [Chartify](https://github.com/spotify/chartify/) - Same interface for plots with a goal to make it easy to use Bokeh plots.
- [bqplot](https://github.com/bloomberg/bqplot) - 2-D plotting library for Project Jupyter
- [ipyleaflet](https://github.com/jupyter-widgets/ipyleaflet) - A Jupyter / Leaflet bridge enabling interactive maps in the Jupyter notebook.
- [pythreejs](https://github.com/jupyter-widgets/pythreejs) -
  A Python / ThreeJS bridge utilizing the Jupyter widget infrastructure.
- [gmaps](https://jupyter-gmaps.readthedocs.io/en/stable/tutorial.html) - Google Maps For Jupyter Notebooks
- [vaex](https://vaex.io/) - Lazy Out-of-Core DataFrames for Python. Visualize a billion rows per second on a single computer.
- [scattertext](https://github.com/JasonKessler/scattertext) - Beautiful visualizations of how language differs among document types.
- [missingno](https://github.com/ResidentMario/missingno) - Visualize missing data
- [pygal](https://github.com/Kozea/pygal) - Python svg Graph plotting Library
- [geoplotlib](https://github.com/andrea-cuttone/geoplotlib) - python toolbox for visualizing geographical data and making maps
- [gleam](https://github.com/dgrtwo/gleam) - Creating interactive visualizations with Python
- [geonotebook](https://github.com/OpenGeoscience/geonotebook) - A Jupyter notebook extension for geospatial visualization and analysis
- [Bokeh Tutorial (Notebooks)](https://github.com/bokeh/bokeh-notebooks/tree/master/tutorial)
- [Bokeh](https://bokeh.pydata.org/en/latest/docs/user_guide.html#userguide)
- [Responsive Bar Charts with Bokeh, Flask and Python 3](https://www.fullstackpython.com/blog/responsive-bar-charts-bokeh-flask-python-3.html)
- [Dash](https://github.com/plotly/dash) - Analytical Web Apps for Python. No JavaScript Required.
- [Creating Interactive Visualizations with Plotly’s Dash Framework](https://pbpython.com/plotly-dash-intro.html)
- [Superset](https://github.com/apache/incubator-superset) - Apache Superset (incubating) is a modern, enterprise-ready business intelligence web application
- [NetworkX](https://github.com/networkx/networkx) - NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
- [Seaborn](http://seaborn.pydata.org/) - Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
- [ggpy](https://github.com/yhat/ggpy) - ggplot for Python
- [Matplotlib](https://github.com/matplotlib/matplotlib) - matplotlib: plotting with Python
- [Effectively Using Matplotlib](https://pbpython.com/effective-matplotlib.html)
- [Plotly](https://github.com/plotly/plotly.py) - An open-source, interactive graphing library for Python
- [Generate HTML reports with D3 graphs using Python, Plotly, and Pandas](https://nbviewer.jupyter.org/gist/jackparmer/7d27637328a93e6d699b)
- [AnatomyOfMatplotlib (Tutorial)](https://github.com/matplotlib/AnatomyOfMatplotlib) - Anatomy of Matplotlib -- tutorial developed for the SciPy conference
- [pyLDAvis](https://github.com/bmabey/pyLDAvis) - Python library for interactive topic model visualization. Port of the R LDAvis package
- [DataShader](http://datashader.org/) - uses Dask and Numba to visualize huge data sets
- [Graph-Tool](https://graph-tool.skewed.de/) - Graph-tool is an efficient Python module for manipulation and statistical analysis of graphs (a.k.a. networks)
- [Vispy](https://github.com/vispy/vispy)
- [Panel](https://github.com/pyviz/panel) - A high-level Python toolkit for composing widgets and plots https://panel.pyviz.org
- [jupyter-matplotlib](https://github.com/matplotlib/jupyter-matplotlib/) - Matplotlib Jupyter Extension
- [ipysheet](https://github.com/QuantStack/ipysheet) - Jupyter handsontable integration
- [ipywebrtc](https://github.com/maartenbreddels/ipywebrtc) - WebRTC for Jupyter notebook/lab
- [weasyprint](https://github.com/Kozea/WeasyPrint) - WeasyPrint converts web documents (HTML with CSS, SVG, …) to PDF. https://weasyprint.org/ ([Creating PDF Reports with Pandas, Jinja and WeasyPrint](https://pbpython.com/pdf-reports.html))
- [PySal](https://github.com/pysal/pysal) - PySAL: Python Spatial Analysis Library http://pysal.org

#### Data

- [Our World in Data](https://ourworldindata.org/) - data visualizations of many things around the world

---

#### Monte Carlo Simulations

- [PBP - Monte Carlo Simulation with Python](https://pbpython.com/monte-carlo.html) - ([Code](https://github.com/chris1610/pbpython/blob/master/notebooks/Monte_Carlo_Simulationv2.ipynb))
- [Monte Carlo simulation in Python](https://www.mikulskibartosz.name/monte-carlo-simulation-in-python/)

#### Data Pipeline Tools

- [Pandas](https://github.com/pandas-dev/pandas) - Flexible and powerful data analysis / manipulation library for Python, providing labeled data structures similar to R data.frame objects, statistical functions, and much more
- [Modin](https://github.com/modin-project/modin) - Modin: Speed up your Pandas workflows by changing a single line of code
- [Pandas-Profiling](https://github.com/pandas-profiling/pandas-profiling) - Create HTML profiling reports from pandas DataFrame objects
- [Dask](https://github.com/dask/dask) - Parallel computing with task scheduling
- [ray](https://github.com/ray-project/ray) - A system for parallel and distributed Python that unifies the ML ecosystem.
- [Spark](https://spark.apache.org/docs/latest/index.html) - [PySpark API](https://spark.apache.org/docs/latest/api/python/index.html#)
- [Optimus](https://github.com/ironmussa/Optimus) - Agile Data Science Workflows made easy with Pyspark https://hioptimus.com
- [pypeln](https://github.com/cgarciae/pypeln)- Concurrent data pipelines made easy
- [smart_open](https://github.com/RaRe-Technologies/smart_open) - Utils for streaming large files (S3, HDFS, gzip, bz2...)
- [Blaze](https://github.com/blaze/blaze) - NumPy and Pandas interface to Big Data
- [Faker](https://github.com/joke2k/faker) - Faker is a Python package that generates fake data for you.
- [Kedro](https://github.com/quantumblacklabs/kedro) - A Python library for building robust production-ready data and analytics pipelines

---

#### Time Series

- [Prophet](https://facebook.github.io/prophet/) - A Facebook Time Series Analysis library
- [PyFlux](https://pyflux.readthedocs.io/en/latest/index.html) - Time series analysis library with flexible range of modelling and inference options.

---

#### NLP/Text Manipulation

- [spaCy](https://github.com/explosion/spaCy) - Industrial-strength Natural Language Processing (NLP) with Python and Cython
- [FlashText](https://flashtext.readthedocs.io/en/latest/) - This module can be used to replace keywords in sentences or extract keywords from sentences.
- [textacy](https://github.com/chartbeat-labs/textacy) - NLP, before and after spaCy
- [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy) - Fuzzy string matching in Python
- [re](regex)
- [pampy](https://github.com/santinic/pampy)
- [textblob](https://github.com/sloria/textblob) - Simple, Pythonic, text processing--Sentiment analysis, part-of-speech tagging, noun phrase extraction, translation, and more.
- [nlp-text-mining-working-examples](https://github.com/kavgan/nlp-text-mining-working-examples)
- [Pattern](https://github.com/clips/pattern) - Web mining module for Python, with tools for scraping, natural language processing, machine learning, network analysis and visualization
- [Feature Engineering Text Data - Traditional Strategies](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%20content/feature%20engineering%20text%20data/Feature%20Engineering%20Text%20Data%20-%20Traditional%20Strategies.ipynb)
- [Feature Engineering Text Data - Advanced Deep Learning Strategies](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%20content/feature%20engineering%20text%20data/Feature%20Engineering%20Text%20Data%20-%20Advanced%20Deep%20Learning%20Strategies.ipynb)
- [NLP Strategy I - Processing and Understanding Text](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%20content/nlp%20proven%20approach/NLP%20Strategy%20I%20-%20Processing%20and%20Understanding%20Text.ipynb)
- [Practical Text Classification With Python and Keras](https://realpython.com/python-keras-text-classification/)
- [Regex tutorial — A quick cheatsheet by examples](https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285)
- [spaCy](https://github.com/explosion/spaCy) - Industrial-strength Natural Language Processing (NLP) with Python and Cython https://spacy.io
- [xlnet](https://github.com/zihangdai/xlnet) - XLNet: Generalized Autoregressive Pretraining for Language Understanding
- [xlnet-Pytorch](https://github.com/graykode/xlnet-Pytorch) - Simple XLNet implementation with Pytorch Wrapper https://arxiv.org/pdf/1906.08237.pdf

---

#### Deep Learning Tools

- [Pyro](http://pyro.ai/examples/) - Pyro is a flexible, scalable deep probabilistic programming library built on PyTorch
- [allennlp](https://github.com/allenai/allennlp) - An open-source NLP research library, built on PyTorch.
- [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP) - Supporting Rapid Prototyping with a Toolkit (incl. Datasets and Neural Network Layers)
- [OpenCV-Python](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html) - Open Source Computer Vision Library for Python
- [Screenshot-to-code](https://github.com/emilwallner/Screenshot-to-code) - A neural network that transforms a design mock-up into a static website
- [TensorFlow Models Examples](https://github.com/tensorflow/models/tree/master/official)
- [fastai](https://github.com/fastai/fastai) - The fastai deep learning library, plus lessons and and tutorials
- [Keras](https://github.com/keras-team/keras)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [keras-tuner](https://github.com/keras-team/keras-tuner) - Hyperparameter tuning for humans
- [Visdom](https://github.com/facebookresearch/visdom) - A flexible tool for creating, organizing, and sharing visualizations of live, rich data. Supports Torch and Numpy.
- [talos](https://github.com/autonomio/talos) - Hyperparameter Optimization for Keras Models
- [keras-contrib](https://github.com/keras-team/keras-contrib)

#### Recommender Systems

- [list_of_recommender_systems](https://github.com/grahamjenson/list_of_recommender_systems)
- [Amazon-Product-Recommender-System](https://github.com/mandeep147/Amazon-Product-Recommender-System) - Sentiment analysis on Amazon Review Dataset available at http://snap.stanford.edu/data/web-Amazon.html
- [Building a Recommendation System Using Neural Network Embeddings](https://towardsdatascience.com/building-a-recommendation-system-using-neural-network-embeddings-1ef92e5c80c9)
- [Neural Network Embeddings Explained](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526)

#### Web Scraping Tools

- [BeautifulSoup]()
- [requests]()
- [MechanicalSoup](https://github.com/MechanicalSoup/MechanicalSoup) - A Python library for automating interaction with websites.

#### SDLC Tools

- [yapf](https://github.com/google/yapf) - yet another python formatter
- [black](https://github.com/ambv/black) - The uncompromising Python code formatter
- [precommit](https://github.com/pre-commit/pre-commit) ([article](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)) - A framework for managing and maintaining multi-language pre-commit hooks. (Black + Flake8)

#### Others

- [wget](https://bitbucket.org/techtonik/python-wget/src) - free utility for non-interactive downloading files from the web
- [pendulum](https://github.com/sdispater/pendulum) - Python datetime manipulation made easy
- [python-dotenv](https://github.com/theskumar/python-dotenv) - Get and set values in your .env file in local and production servers.
- [sqlparse](https://github.com/andialbrecht/sqlparse) - sql parsing tool
- [credstash](https://github.com/fugue/credstash) - A little utility for managing credentials in the cloud

#### Database Connectors

- [graphene](https://github.com/graphql-python/graphene) - GraphQL framework for Python

#### Pub-Sub, Message Queues, Streaming

- [kq](https://github.com/joowani/kq) - Kafka-based Job Queue for Python
- [sockets]()
- [pykafka](https://github.com/Parsely/pykafka) - Apache Kafka client for Python; high-level & low-level consumer/producer, with great performance.
- [awesome-kafka](https://github.com/infoslack/awesome-kafka) - A list about Apache Kafka
- [How to Stream Text Data from Twitch with Sockets in Python](https://learndatasci.com/tutorials/how-stream-text-data-twitch-sockets-python/)

#### Web Frameworks [Some Benchmarks](https://github.com/the-benchmarker/web-frameworks)

- [Pyramid](https://github.com/Pylons/pyramid) - A Python web framework https://trypyramid.com/
- [sanic](https://github.com/huge-success/sanic) - Async Python 3.5+ web server that's written to go fast
- [Tornado](https://github.com/tornadoweb/tornado) - Tornado is a Python web framework and asynchronous networking library, originally developed at FriendFeed.
- [Falcon](https://github.com/falconry/falcon) - Falcon is a bare-metal Python web API framework for building high-performance microservices, app backends, and higher-level frameworks. [Docs](https://falconframework.org/)
- [Vibora](https://github.com/vibora-io/vibora) - Fast, asynchronous and elegant Python web framework.
- [japronto](https://github.com/squeaky-pl/japronto) - Screaming-fast Python 3.5+ HTTP toolkit integrated with pipelining HTTP server based on uvloop and picohttpparser.
- [aiohttp](https://github.com/aio-libs/aiohttp) - Asynchronous HTTP client/server framework for asyncio and Python https://docs.aiohttp.org
- [fastapi](https://github.com/tiangolo/fastapi), [Docs](https://fastapi.tiangolo.com/) - FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.

#### Twitter APIs
- [twint](https://github.com/twintproject/twint)

#### Networking Tools

- [uvloop](https://github.com/MagicStack/uvloop) ([article](https://magic.io/blog/uvloop-blazing-fast-python-networking/)) - Ultra fast asyncio event loop (faster than NodeJS, close to Go speed)

#### Flask Tools

- [Flask-AppBuilder](https://github.com/dpgaspar/Flask-AppBuilder) - Simple and rapid application development framework, built on top of Flask. includes detailed security, auto CRUD generation for your models, google charts and much more.
- [Awesome-Flask](https://github.com/humiaozuzu/awesome-flask) - A curated list of awesome Flask resources and plugins

#### Static File Generators

- [Nikola](https://github.com/getnikola/nikola) - A static website and blog generator
- [Pelican](https://github.com/getpelican/pelican) - Static site generator that supports Markdown and reST syntax. Powered by Python.

#### PDF Converters

- [python-pdfkit](https://github.com/JazzCore/python-pdfkit) - Wkhtmltopdf python wrapper to convert html to pdf
- [pdftabextract](https://github.com/WZBSocialScienceCenter/pdftabextract) - A set of tools for extracting tables from PDF files helping to do data mining on (OCR-processed) scanned documents.

#### Documentation Libraries

- Sphinx
- [pweave](http://mpastell.com/pweave/) - Pweave is a scientific report generator and a literate programming tool for Python. It can capture the results and plots from data analysis and works well with numpy, scipy and matplotlib.

---

#### D3

- [Hitchhiker's Guide to D3.js](https://medium.com/@enjalot/the-hitchhikers-guide-to-d3-js-a8552174733a)
- [D3 Charts and Chartbuilder](https://blockbuilder.org/search)
- [D3 Tutorials](https://github.com/d3/d3/wiki/Tutorials)
- [D3 API Reference](https://github.com/d3/d3/blob/master/API.md)
- [Awesome-D3](https://github.com/wbkd/awesome-d3#charts)
- [Data Visualization with D3.js, a FreeCodeCamp course](https://www.youtube.com/watch?v=_8V5o2UHG0E)

#### APIs

- [Tweepy](https://github.com/tweepy/tweepy)

#### Hacking, PCAPs, and Network Analysis

- [Sublist3r](https://github.com/aboul3la/Sublist3r)
- [scapy](https://github.com/secdev/scapy)
- [knock](https://github.com/guelfoweb/knock) - subdomain scanner
- [scapy-http](https://github.com/invernizzi/scapy-http)
- [dpkt](https://github.com/kbandla/dpkt)
- [kamene](https://github.com/phaethon/kamene)
- [pcapy](https://github.com/SecureAuthCorp/pcapy) - Pcapy is a Python extension module that interfaces with the libpcap packet capture library.
- [pyshark](https://github.com/KimiNewt/pyshark) - Python wrapper for tshark, allowing python packet parsing using wireshark dissectors
- [PyPCAPKit](https://github.com/JarryShaw/PyPCAPKit) - Python multi-engine PCAP analyse kit.
- [fsociety](https://github.com/Manisso/fsociety) - fsociety Hacking Tools Pack – A Penetration Testing Framework
- [PayloadsAllTheThings](https://github.com/swisskyrepo/PayloadsAllTheThings) - A list of useful payloads and bypass for Web Application Security and Pentest/CTF

#### GUIs

- [PySimpleGUI](https://github.com/MikeTheWatchGuy/PySimpleGUI)

#### Repeatable Python Workflows in Notebooks

- [Building a Repeatable Data Analysis Process with Jupyter Notebooks](http://pbpython.com/notebook-process.html)

#### Prototyping Projects

- [Learn to Build Machine Learning Services, Prototype Real Applications, and Deploy your Work to Users](https://towardsdatascience.com/learn-to-build-machine-learning-services-prototype-real-applications-and-deploy-your-work-to-aa97b2b09e0c)
- [Combining D3 with Kedion: Graduating from Toy Visuals to Real Applications](https://towardsdatascience.com/combining-d3-with-kedion-graduating-from-toy-visuals-to-real-applications-92bf7c3cc713)
- [Learning How to Build a Web Application](https://medium.com/@rchang/learning-how-to-build-a-web-application-c5499bd15c8f)

#### Machine Learning - Interesting Repos

- [Siraj Raval](https://github.com/llSourcell)
  https://github.com/rushter/MLAlgorithms
  https://github.com/WillKoehrsen/Data-Analysis
  https://github.com/WillKoehrsen/machine-learning-project-walkthrough
  https://github.com/Avik-Jain/100-Days-Of-ML-Code
  https://github.com/llSourcell/Learn_Data_Science_in_3_Months
  https://github.com/llSourcell/Learn_Machine_Learning_in_3_Months
  https://github.com/llSourcell/100_Days_of_ML_Code
  https://github.com/ZuzooVn/machine-learning-for-software-engineers
  https://github.com/llSourcell/Learn_Deep_Learning_in_6_Weeks
  https://github.com/Spandan-Madan/DeepLearningProject
- [Machine Learning from Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
- [Jupyter notebooks from the scikit-learn video series - Justin Markham](https://github.com/justmarkham/scikit-learn-videos)
- [fast.ai](https://www.fast.ai/) - The fastai deep learning library, plus lessons and and tutorials http://docs.fast.ai
- [Coursera ML - Andrew Ng in Python](https://github.com/JWarmenhoven/Coursera-Machine-Learning)
- [Think Python 2nd ed](https://github.com/AllenDowney/ThinkPython2)
- [Thinks Stats 2nd ed](https://github.com/AllenDowney/ThinkStats2)
- [Think Bayes](https://github.com/AllenDowney/ThinkBayes2)
- [Make Your Own Neural Network](https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork)
- [Learning AI if you Suck at Math](https://hackernoon.com/learning-ai-if-you-suck-at-math-8bdfb4b79037)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
- [Introduction to Statistics With Python](https://github.com/thomas-haslwanter/statsintro_python)

#### ML Articles

- [How To Create Data Products That Are Magical Using Sequence-to-Sequence Models](https://towardsdatascience.com/how-to-create-data-products-that-are-magical-using-sequence-to-sequence-models-703f86a231f8)
- [How To Create Natural Language Semantic Search For Arbitrary Objects With Deep Learning](https://towardsdatascience.com/semantic-code-search-3cd6d244a39c)
- [Machine Learning for Everyone](https://vas3k.com/blog/machine_learning/)

#### Docker Articles

- [Docker articles by Docker](https://docker-software-inc.scoop.it/t/docker-by-docker)

#### Git Commands

- [Git-Commands](https://github.com/joshnh/Git-Commands)

#### Codelabs

- https://codelabs.developers.google.com/?cat=tensorflow

#### Awesome Links

- [Awesome-BigData](https://github.com/onurakpolat/awesome-bigdata)
- [Awesome-Hacking-Resources](https://github.com/vitalysim/Awesome-Hacking-Resources)
- [awesome-web-hacking](https://github.com/infoslack/awesome-web-hacking)
- [awesome-data-engineering](https://github.com/igorbarinov/awesome-data-engineering)
- [awesome-elasticsearch](https://github.com/dzharii/awesome-elasticsearch)
- [awesome-datascience](https://github.com/bulutyazilim/awesome-datascience)
- [awesome-flask](https://github.com/humiaozuzu/awesome-flask#full-text-searching)

#### NBA Shot Charts

- [nbashots](https://github.com/savvastj/nbashots) - NBA shot charts using matplotlib, seaborn, and bokeh.
- [nba-team-shot-charts](https://danielwelch.github.io/nba-team-shot-charts.html)

---

## Courses

#### Deep Learning Courses

- [Fast.ai](https://course.fast.ai/index.html)

---

#### More Random Packages

- [Data Science Blogs](https://github.com/rushter/data-science-blogs)
- [Drag and Drop JS](https://github.com/SortableJS/Sortable)
- [Projects](https://github.com/karan/Projects/) - [Project Solutions](https://github.com/karan/Projects-Solutions)
- [project-based-learning#python](https://github.com/tuvtran/project-based-learning#python)

#### Notebook tools -- Older, less maintained

- [ipyaggrid](https://dgothrek.gitlab.io/ipyaggrid/) ([Features](https://www.ag-grid.com/javascript-grid-features/)) ([Article](https://medium.com/@olivier.borderies/harnessing-the-power-of-ag-grid-in-jupyter-3ae27fb21012)) - Using ag-Grid in Jupyter notebooks
- [PixieDust](https://github.com/pixiedust/pixiedust) ([YouTube Example](https://www.youtube.com/watch?v=FoOHFlkCaXI)) ([DeBugger article](https://medium.com/ibm-watson-data-lab/the-visual-python-debugger-for-jupyter-notebooks-youve-always-wanted-761713babc62)) - [PixieDust 1.0 is here (blog)](https://medium.com/ibm-watson-data-lab/pixiedust-1-0-is-here-15e0f428df88) - Easy Data Visualizer, Debugger, etc for Notebooks
- [nbdime](https://github.com/jupyter/nbdime) - Tools for diffing and merging of Jupyter notebooks
- [nteract](https://github.com/nteract/nteract) - The interactive computing suite for you!
- [Sparkmagic](https://github.com/jupyter-incubator/sparkmagic) - Jupyter magics and kernels for working with remote Spark clusters
- [PayPal Notebooks, powered by Jupyter: Enabling the next generation of data scientists at scale](https://medium.com/paypal-engineering/paypal-notebooks-powered-by-jupyter-fd0067bd00b0)
- [PayPal Notebook Extensions](https://github.com/paypal/PPExtensions)
- [nbtutor](https://github.com/lgpage/nbtutor) - Visualize Python code execution (line-by-line) in Jupyter Notebook cells.
- [py_d3](https://github.com/ResidentMario/py_d3) - D3 block magic for Jupyter notebook
- [Jupyter Notebook Formatting Extensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions)
- [Jupyter Notebooks Dashboards Extension](https://jupyter-dashboards-layout.readthedocs.io/en/latest/using.html)
- [JupyterLab-toc](https://github.com/jupyterlab/jupyterlab-toc) - Table of Contents Extension
- [Jupyter Extension Tricks](https://codeburst.io/jupyter-notebook-tricks-for-data-science-that-enhance-your-efficiency-95f98d3adee4)
- [Bringing the best out of Jupyter Notebooks for Data Science](https://towardsdatascience.com/bringing-the-best-out-of-jupyter-notebooks-for-data-science-f0871519ca29)
- [Airbnb - Knowledge Repo](https://github.com/airbnb/knowledge-repo) - tag and share analysis
- [Advanced Jupyter Notebooks: A Tutorial](https://www.dataquest.io/blog/advanced-jupyter-notebooks-tutorial/)
- [imgkit](https://github.com/jarrekk/imgkit) - Wkhtmltoimage python wrapper to convert html to image
