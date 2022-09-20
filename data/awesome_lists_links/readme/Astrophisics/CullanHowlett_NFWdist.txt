# NFWdist
This module lets you sample from the NFW as a true PDF with the only variable being the concentration (con in the package).

# Load

Load with:
```
from NFWdist import *
```

# Checks
Run:
```
python test.py
```
for some simple timing checks compared to the Uniform distribution and to see that that our random draws (using the rnfw via the qnfw function) produces the right PDFs (as per dnfw):

