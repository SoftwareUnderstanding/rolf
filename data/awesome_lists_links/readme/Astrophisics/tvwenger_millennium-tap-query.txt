# millennium-tap-query

## UPDATE: This tool no loner works due to the discontinuation of the Millennium TAP Web Client.

A Python Tool to Query the Millennium Simulation UWS/TAP client

This is a simple wrapper for the Python package `requests` to deal
with connections to the [Millennium TAP Web
Client](http://galformod.mpa-garching.mpg.de/millenniumtap/).  With
this tool you can perform basic or advanced queries to the Millennium
Simulation database and download the data products.

This tool is similar to the [TAP
query](http://svn.ari.uni-heidelberg.de/svn/gavo/python/trunk/docs/tapquery.rstx)
tool in the German Astrophysical Virtual Observatory (GAVO) VOtables
package.

## Citation

If you use this tool in any published work, please use the following reference: http://dx.doi.org/10.5281/zenodo.47429

## Prerequisites

This program is tested on both Python 2.7 and Python 3.5. You can
install the required prerequisites using `pip` via
```bash
pip install -r requirements.txt
```

You will also need access (username and password) to the Millennium Simulation database. You can register for access [here](http://galformod.mpa-garching.mpg.de/portal/contact.html).

## Usage Examples
```python
"""
Set up a new job, perform a simple query, and save the results
"""
import millennium_query
username='foo' # replace with your login credentials
password='bar'
conn = millennium_query.MillenniumQuery(username,password)
conn.query('SELECT TOP 10 * FROM MPAGalaxies..DeLucia2006a')
conn.run() # Start job
conn.wait() # Wait until job is finished
conn.save('results.csv') # Stream results to file
conn.delete() # Delete job from server
conn.close() # close connection
```

```python
"""
Connect to an already submitted job and save the results
"""
job_id = 'foobar'
conn = millennium_query.MillenniumQuery(username,password,job_id=job_id)
conn.wait() # Make sure job is done
conn.save('results.csv')
conn.delete()
conn.close()
```

```python
"""
Acquire cookies, then run several jobs
"""
cookies = millennium_query.get_cookies(username,password)
jobs = []
for i in range(10):
    conn = millennium_query.MillenniumQuery(username,password,cookies=cookies)
    conn.query('SELECT TOP {0} * FROM MPAGalaxies..DeLucia2006a'.format(i))
    conn.run()
    jobs.append(conn)
for i,conn in enumerate(jobs):
    conn.wait()
    conn.save('results_{0}.csv'.format(i))
    conn.delete()
    conn.close()
```

## Details

This tool contains two standalone functions and the `MillenniumQuery`
class.

`get_response` performs the actual query to the TAP client. It takes
two arguments: `session` is a `requests.Session()` object and `url` is
the URL to query. The keyword arguments are `method` (default is
`'GET'`) determines whether the query should use GET or POST, `data`
(default is `None`) is the POST or GET data, `cookies` (default is
`None`), `max_attempts` (default is `5`), and stream (default is
`False`). This function expects `session.auth` to be appropriately set
already. It will perform the query up to `max_attempts` times before
giving up and raising a `RuntimeError`. It will also raise a
`RuntimeError` if it does not get the expected XML webpage (usually a
sign that the login credentials are incorrect). If successful, it
returns the `response` returned by the call to `session.get()` or
`session.post()`.

`get_cookies` performs a simple connect to download site cookies.  It
takes two arguments: `username` and `password`, your login credentials
to the Millennium TAP client. The function returns a
`RequestsCookieJar` object.

The constructor for the `MillenniumQuery` object also takes the
`username` and `password` arguments, as well as the following keyword
arguments: `job_id` (default is `None`) is set if you wish to connect
to a previously submitted job, `query_lang` (default is `SQL`) to set
the query language, `results_format` (default is `csv`) sets the
TAP-returned filetype, `max_rec` (default is `100000`) is the maximum
number of records to retrieve, and `cookies` (default is None). For
more details on these parameters, see the [MillenniumTAP help
page](http://galformod.mpa-garching.mpg.de/millenniumtap/tapface/pages/help.jsp).
The constructor sets up the `requests.Session()` object and obtains
the necessary cookies.

`MillenniumQuery:query` takes one parameter, `query`. This query is
sent to the TAP client. If no `job_id` was specified in the
constructor, this function determines the `job_id` for this job.

`MillenniumQuery:run` starts any queried job on the TAP client. 
This function will raise `RuntimeError` if the `query` function has
not already been called.

`MillenniumQuery:wait` holds execution until the TAP client returns
a `COMPLETED`, `ABORTED`, or `ERROR` status. The second two statuses
will raise a `RuntimeError`. This function takes a keyword argument
`max_attempts` (default is 100) that determines the maximum number of
`while` iterations before raising a `RuntimeError`. The pause between
each successive iteration grows exponential to a maximum of 2 minutes.

`MillenniumQuery:save` streams the results file and saves to the a
file defined by the argument `filename`. 

`MillenniumQuery:close` closes the session.

## Warranty and Copyright

Copyright (C) 2016 by Trey Wenger: tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Contribute

If you find a bug, please submit an issue and I will try to fix it
as soon as possible.

If you would like to contribute to the development of this tool,
feel free to fork and submit pull requests.
