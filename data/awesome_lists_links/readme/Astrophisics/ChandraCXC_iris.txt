#Iris: the VO SED Analysis application

Latest stable release: 3.0b1 - Aug, 17th, 2016.

Iris is a Virtual Observatory (VO)
application for analysis of spectral energy distributions (SEDs).  Iris can
retrieve SEDs from the NASA Extragalactic Database (NED) or read the user's SED
data from file.  Iris can build and display SEDs, allow the user to select
particular data points for analysis, and fit models to SEDs.

Iris can also retrieve photometric data from the Italian Space Agency Science
Data Center (ASDC) and from the Centre de Donnees astronomiques (CDS).

The components of Iris were originally developed by members of the Virtual
Astronomical Observatory (VAO).
NED is a service provided by IPAC at Caltech.
The Graphical User Interface, as well as Sherpa, the fitting engine,
are developed by the Chandra project at the Harvard-Smithsonian
Center for Astrophysics: Sherpa provides a library of models, fit statistics,
and optimization methods for modeling SEDs.

SED Builder allows to build SED instances combining data from several sources.

Communication between the Java GUI and Sherpa is managed by a SAMP
connection. SAMP is an International Virtual Observatory Alliance standard
for process intercommunication.

Iris provides interoperability features in input (import data from other VO-
enabled applications) and output (export an SED to VO-enabled applications).

# How to Install Iris

If you installed a previous version of Iris,
we suggest you install Iris in a different `conda` environment.
However, you can simply run `conda update iris` if you
want to override the existing installation.

To install Iris in a new environment:
````
$ conda create -n iris3 -c cxc -c sherpa iris=3.0b1
````

The `-c cxc -c sherpa` is not necessary if you have already these CXC channels
listed in your `$HOME/.condarc` file.

You can run Iris with:
````
$ source activate iris3
$ iris
````

You can also run the smoke test in order to verify your installation is working
properly:
````
$ iris smoketest
````

# Source Build
Iris is the combination of several Python and Java packages. Source builds are
thus more complex than the usual processes for individual packages alone.

The following instructions assume you have `conda` installed.
Conda is part of the Anaconda distribution and can be easily installed
through the Miniconda minimal distribution.

````
$ git clone --recursive https://github.com/ChandraCXC/iris
$ conda create -n iris python=2.7 astropy=0.4.4 scipy
$ source activate iris
$ conda install -c sherpa sherpa
$ pip install sampy
$ pip install astlib
$ cd sherpa-samp; python setup.py develop; cd ..
$ cd sedstacker; python setup.py develop; cd ..
````

You should also make sure that `sherpa-samp` is working.
After installing `sherpa-samp`, run `sherpa-samp` from the
command line. The program should start and listen for SAMP
connections. After a while the program times out and exits.
It's important that `sherpa-samp` does not exit with errors.

You can also run the Iris smoke test by:

````
$ bash iris/target/Iris smoketest
````

## How to run the unit and integration tests

### Without coverage analysis

````
$ mvn clean test # Unit tests
$ mvn clean test-compile failsafe:integration-test # Integration tests only
$ mvn clean verify # All tests
````

### With JaCoCo coverage analysis

````
$ mvn -Pjacoco test # Unit tests
$ mvn -Pjacoco verify # All tests
$ mvn -Pjacoco jacoco:report # generate report
````

Note that individual reports will be created in each individual submodule.

### With Sonar

A [[http://www.sonarqube.org/ | SonarQube]] instance must be running.
The configuration for the SonarQube instance must be placed into
the maven local `settings.xml` file for connecting with the database
backing the SonarQube instance.

````
$ mvn -Psonar install # All tests
$ mvn sonar:sonar
````

# 3.0b1 Release Notes

Iris 3.0b1 introduces a new infrastructure for visualizing and fitting
spectrophotometric data. The new infrastructure allows users to load and fit
high resolution spectra as well as broadband, multi-wavelength
spectrophotometric datasets. The new infrastructure is also more flexible and
it will enable future extensions of the Iris functionality. Moreover, Iris 3.0b1
introduces a simple client for the Vizier/CDS SED service while keeping the
dedicated clients to NED and ASDC services. This release also fixes several bugs
and introduces some new functionality and user interface improvements, as
specified in more detail below, component by component.

## Fitting Tool

The fitting tool GUI has been completely redesigned. It now relies on a single
window, and information should be easier to set and retrieve:

    - The list of available models is always visible: you can double-click
on a model to make it part of your model expression.
    - A search box allows to easily filter the model components,
and a description box displays simple documentation for the model.
    - We introduced a simple "Chi2" statistic (In Iris 2.1 you had to select one
of the specialized chi2 statistics and Sherpa would fall back to Chi2 if errors
were provided by the user).
    - Model parameter values are updated on the fly.
    - You can select ranges by either clicking on the plot or by manually
setting ranges in any units.
    - Model components now have unique IDs in a working session. Component
IDs do not change if components are deleted.
    - Model expressions are validated on the fly, so you know if a model
expression is not valid as soon as you type it.
    - Output files created by the "Save Text" option now contain more
information, including the location of custom user models.
    - Models can be saved as Json files and then loaded back into Iris.
Note that the Iris 1 and 2 xml serializations are not supported any more. If you
have such a file and you want it converted to the new format,
please let us know.
    - Models can be evaluated even if they have not been fitted.
So for instance you can change model parameters and re-evaluate the model,
or evaluate individual model components.
    - You can right-click on a model component to select it and remove it
from the expression.

## SED Viewer

The viewer component was completely redesigned. It now relies on STILTS as a
plotting backend. Functionality is mostly unchanged, but now you can:

    - plot and analyze high resolution spectra.
    - coplot SEDs with their models.
    - plot model functions even if they have not been fitted. You can also plot
individual model components.

Note that SED with multiple segments show points belonging to different
segments with different colors. The current palette has 16 distinct colors.
More than 16 segments would result in points having color differences that are
hardly noticeable, not to mention a rather long legend. When more than 16
segments are plotted, Iris will show the SED as a single segment.
The Metadata Browser is unaffected by the number of segments.

## Metadata Browser

The metadata browser GUI, which is accessible by clicking "Metadata" on the
Visualizer toolbar, has been completely redesigned. Functionality is mostly
unchanged, however:

    - Points can now be masked and unmasked directly from the metadata browser.
    - Masked points are not included in the fit.

## Sed Stacker

The GUI was redesigned to follow suggestions from users. The SED Stacker frame
is now laid out in a horizontal fashion, so users will move from left to right
on the frame as they work on a Stack. Open Stacks and Stack Management are on
the left; a list of added SEDs (with Add/Remove capabilities) is in the center;
redshifting, normalizing, and stacking options are on the far right.
Right-clicking a Stack in the Open Stacks window now offers a Remove option.
We also fixed some typos in the GUI.

# Caveats and Known Bugs

## Fitting Tool

    - The tool is currently missing "bulk" operations on model parameters
(e.g. thaw all, freeze all).
    - There is currently no warning when overwriting existing files.
- Visualizer:
    - Residuals and main plot are not bound together when zooming, panning, etc.
    - Depending on the size of the Visualizer window, legend might spill over
the bounds of the Visualizer when many segments are displayed.
Workaround: users can hide the legend from the "View" menu.
    - It is not possible to filter data points by selecting them
in the Visualizer.
    - When analyzing SEDs with more than 16 segments, fitting ranges are not
visualized in the plotter. However, they are still listed in the
Fitting Ranges window.

## Metadata Browser

    - Filtering data by filter expressions has been completely redesigned and
is much more responsive. However, it only applies to numerical columns.
Also, when masking points, a new column is added and column identifiers change.
At this time, scientific notation is not fully supported,
especially with negative exponents, e.g. 1e-5.
    - Simple scaling ("aperture correction") has been disabled.
Future versions of Iris will provide a mechanism for performing
arbitrary operations on columns.
    - SAMP broadcasting is not available any more from the Metadata Browser
directly. One can extract an SED in the Metadata Browser and then use the SED
Builder capabilities to broadcast a flattened SED or an arbitrary
number of segments.
    - Data can now be sorted only according to one column, not two
as in Iris 2.1.

## Sed Stacker

    - After stacking a group of SEDs, the resultant SED is added to the
SED Builder. A silent java.lang.NullPointerException exception is raised when
a user tries to add new segments to this SED. No warning pops up to the user.
    - Sometimes the normalization configuration parameters don't update
correctly when you switch between Stacks.

