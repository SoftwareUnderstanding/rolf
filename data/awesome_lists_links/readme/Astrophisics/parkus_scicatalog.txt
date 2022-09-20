scicatalog
==========

A module containing a single class for handling catalogs of scientific data in a way that is easily extensible.

Probably the the most valuable aspect of the module is the ability to create nicely formatted AASTex deluxe tables for use in AAS (ApJ, AJ, ...) publications. Sadly, I have not documented these features yet, but you can probably figure them out if you look at the source and save yourself some time!

Currently (2015/07/09) the SciCatalog class handles catalogs of values, their positive and negative uncertainties, and references for those values with methods for easily adding columns and changing values. The catalog is also backed up every time it is loaded under the assumption that it is about to be modified. 

Functionality is pretty minimal at the moment. I created this just to be able to record property of stars that I study.

SciCatalogs are not intended to handle large or even moderately sized databases. Specifically, I have prioritized preserving data with copious backup and disk-writing over speed.

Written by Parke Loyd, 2015/07.

Example
-------
Here's how you'd go about creating a SciCatalog if you already have your data:

    >>> import scicatalog as sc
    >>> from numpy.random import rand
    
    >>> # indices for rows (e.g. names of stars or whatever)
    >>> index = ['thing1', 'thing2']
    
    >>> # column names
    >>> columns = ['col1', 'col2', 'col3']
    
    >>> # abbreviated references and a dictionary defining the abbreviations
    >>> refs = ['a', 'b', 'c', 'd', 'e', 'f']
    >>> ref_definitions = ['blah_' + s for s in refs]
    >>> refDict = dict(zip(refs, ref_definitions))
    >>> refs = [refs[:3], refs[3:]]
    
    >>> # create the catalog. This creates a directory called 'cat' in the present
    >>> # working directory and puts a series of human-readable files recording
    >>> # all the data in there.
    >>> cat = sc.SciCatalog('cat', values=rand(2,3), errpos=rand(2,3), errneg=rand(2,3), refs=refs, refDict=refDict, index=index, columns=columns)
    
    
    >>> # see what's in it with the values, errpos, errneg, and refs attributes
    
    >>> cat.values
                col1      col2      col3
    thing1  0.460304  0.358929  0.205232
    thing2  0.156005  0.841907  0.329851
    
    >>> cat.errpos
                col1      col2      col3
    thing1  0.150619  0.522935  0.305828
    thing2  0.082904  0.501057  0.036864
    
    >>> cat.refs
           col1 col2 col3
    thing1    a    b    c
    thing2    d    e    f
    
    >>> cat.refDict
    {'a': 'blah_a',
     'b': 'blah_b',
     'c': 'blah_c',
     'd': 'blah_d',
     'e': 'blah_e',
     'f': 'blah_f'}
    
    >>> # change one of the values by giving the new value, errors, and reference
    >>> # all at once
    >>> cat.set('thing2', 'col2', 10.0, 2.0, 1.0, 'g')
    UserWarning: The reference key g is not in the reference dictionary for this catalog. You can add it with the `addRefEntry` method.
      "You can add it with the `addRefEntry` method.".format(refkey))
    
    >>> # oops, let's define that reference
    >>> cat.addRefEntry('g', 'blah_g')
    
    >>> # check that the item was updated
    >>> cat.values
                col1       col2      col3
    thing1  0.460304   0.358929  0.205232
    thing2  0.156005  10.000000  0.329851
    >>> cat.errneg
                col1      col2      col3
    thing1  0.668196  0.548406  0.890587
    thing2  0.227836  1.000000  0.048677
    >>> cat.refs
           col1 col2 col3
    thing1    a    b    c
    thing2    d    g    f
    
    
    >>> # add another column, initializes with null values
    >>> cat.addCol('col4')
    >>> cat.values
                col1       col2      col3  col4
    thing1  0.460304   0.358929  0.205232   NaN
    thing2  0.156005  10.000000  0.329851   NaN
    
    >>> cat.refs
           col1 col2 col3  col4
    thing1    a    b    c  none
    thing2    d    g    f  none    
    
    >>> cat.addRow('thing3')
    
    >>> cat.values
                col1       col2      col3  col4
    thing1  0.460304   0.358929  0.205232   NaN
    thing2  0.156005  10.000000  0.329851   NaN
    thing3       NaN        NaN       NaN   NaN
    
    >>> cat.refs
            col1  col2  col3  col4
    thing1     a     b     c  none
    thing2     d     g     f  none
    thing3  none  none  none  none
    
    >>> # let's modify a specific value. Do that using the table attributes
    >>> # which are just pandas DataFrames
    >>> # better make a backup first in case I screw things up
    >>> cat.backup()
    >>> cat.refs['col2']['thing1'] = 'g'
    
    >>> # Gotta save manually now to write that change to the disk.
    >>> cat.save()
    
    
    >>> # let's reload the table from the disk
    >>> del cat
    >>> # load from the disk by just specifying the directory path for the catalog
    >>> cat2 = sc.SciCatalog('cat')
    >>> cat2.values
                col1       col2      col3  col4
    thing1  0.460304   0.358929  0.205232   NaN
    thing2  0.156005  10.000000  0.329851   NaN
    thing3       NaN        NaN       NaN   NaN

Example 2
---------
You're probably more likely to initialize a table and then fill it in as you find the data you need in the scientific literature (or at least that's what I'm doing with stellar properties). This is how that happens.

    >>> import scicatalog as sc
    >>> cat = sc.SciCatalog('cat', columns=['col1', 'col2', 'col3'], index=['thing1', 'thing2'])
    >>> cat.set('thing1', ['col1', 'col2'], value=[1,2], errpos=[3,4])
    >>> cat.set(['thing1', 'thing2'], 'col3', value=[-1, -2], ref=['c', 'd'])
    UserWarning: The reference key c is not in the reference dictionary for this catalog. You can add it with the `addRefEntry` method.
      "You can add it with the `addRefEntry` method.".format(refkey))
    UserWarning: The reference key d is not in the reference dictionary for this catalog. You can add it with the `addRefEntry` method.
      "You can add it with the `addRefEntry` method.".format(refkey))
    
    >>> cat.values
            col1  col2  col3
    thing1     1     2    -1
    thing2   NaN   NaN    -2
    
    >>> cat.errpos
            col1  col2  col3
    thing1     3     4   NaN
    thing2   NaN   NaN   NaN
    
    >>> cat.errneg
            col1  col2  col3
    thing1   NaN   NaN   NaN
    thing2   NaN   NaN   NaN
    
    >>> cat.refs
            col1  col2 col3
    thing1  none  none    c
    thing2  none  none    d
