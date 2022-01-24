# Notes
This is a repository of my notes

#### Here is how you install jupyter nbextensions
```
pip install jupyter_nbextensions_configurator

jupyter nbextensions_configurator enable --user

pip install jupyter contrib_nbextensions

pip install autopep8

pip install npm

pip install nodejs

jupyter contrib nbextension install --user

```

Read below for more useful tools for data-science
https://towardsdatascience.com/bringing-the-best-out-of-jupyter-notebooks-for-data-science-f0871519ca29


```
pip install qgrid
jupyter nbextension enable --py --sys-prefix qgrid
```


#### 6. Embedding URLs, PDFs, and Youtube Videos
```
#Note that http urls will not be displayed. Only https are allowed inside the Iframe
from IPython.display import IFrame
IFrame('https://en.wikipedia.org/wiki/HTTPS', width=800, height=450)
```
![Embedding URLs](https://miro.medium.com/max/658/1*hKNCLc-0g8HubqRZWdWr5Q.gif)

```
from IPython.display import IFrame
IFrame('https://arxiv.org/pdf/1406.2661.pdf', width=800, height=450)
```
![Embedding PDFs](https://miro.medium.com/max/641/1*Trjh8qyP9i0o4Z1LJYp8mg.png)

### Cython
```pip install cython```
![](https://miro.medium.com/max/2483/1*fZS2AARQeqPRyXWEM8DXhg.png)


#### How to create a requirements.txt from your current environment
``` pip freeze > requirements.txt ```
You will then find the requirements.txt file in your current working directory

Then install:
```pip install -r requirements.txt```

Ignore errors when installing packages via requirements.txt
``` FOR /F %p IN (requirements.txt) DO pip install %p ```

https://stackoverflow.com/questions/6457794/pip-install-r-continue-past-installs-that-fail


For Conda install, you can use
```$ FOR /F "delims=~" %f in (requirements.txt) DO conda install --yes "%f" || pip install "%f"```

https://stackoverflow.com/questions/35802939/install-only-available-packages-using-conda-install-yes-file-requirements-t



Exporting Jupyter notebook in HTML 

1.	Run Jupyter notebook and download the notebook in the browser: File->Download as->HTML and you will get a html page with code and output.
2.	Open the exported HTML with browser and activate the browser console with key F12
3.	Run following command in the console:
```document.querySelectorAll("div.input").forEach(function(a){a.remove()})```
4.	The code removes all input div DOM. Then right mouse button and chose "Save Page As" and Save the "Complete page" (not single page).
5.	You will get a page with an associated folder in windows. Use a trick by zip the html page and then extract to unbind the associated. The folder is useless.
6.	Now it is a single html page without code. You can re-distribute it or print it as PDF.

How do you show GIFs in an IPYNB HTML export?
1.	Run this into the cell ```HTML('<IMG SRC="thegif.gif">')```, where "thegif.gif" is your gif's file name
2.	Download your Export into HTML, create a new folder, put the HTML and GIF file in together. 
3.	Create a zip file from the folder and attach it to an email. 
4.	Send it over in an email to Person Bob. 
5.	Bob opens the zip file and drags the folder out to the desktop.
6.	Open the folder that Bob has dragged into the desktop then open your HTML. PS - HTML only works in Chromium browsers. 



How do you show Plotly plots in an IPYNB HTML?
1.	Run your code and have the plot displayed in your .ipynb file
2.	Create a new folder, then download your Export into HTML to that new folder. You will see a new folder created within that folder called "nameofyourfile_files", it has all of the files that you need like your jquery, MathJax, etc..
3.	Create a zip file from the folder and attach it to an email. 
4.	Send it over in an email to Person Bob. 
5.	Bob opens the zip file and drags the folder out to the desktop.
6.	Open the folder that Bob has dragged into the desktop then open your HTML. PS - HTML only works in Chromium browsers. 
