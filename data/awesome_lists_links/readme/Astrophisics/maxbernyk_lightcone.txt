lightcone
=========

light-cone generating script

The script is designed to work with simulated galaxy data stored in a relational
database. Database configuration file is set up for PostgreSQL RDBMS, but can be
modified for use with any other SQL database.

Simulated galaxy data is expexted to be in a box volume. The script re-arranges
the data in a shape of a light-cone.

Included bin2sql.php and bin2sql.py files are sample data import scripts for 
uploading binary simulation data into a PostreSQL database. The light-cone 
constructing script was designed to work with output from the SAGE semi-analytic
model (Croton et al. in prep.), but will work with any other model that has 
galaxy positions (and other properties) saved per snapshots of the simulation 
volume distributed in time.

requirements
============

- PostgreSQL v9.0 or later;
- PHP v5.4 or later;
- php5-pgsql PHP extension;

how to use
==========

Before running the script:

    - set up a PostgreSQL database;
    - adjust database connection and simulation settings in the config.php:
        * database host name;
        * database schema name;
        * database user name;
        * database user password;
        * database connection port;
        * database table name - output catalogues will be saved in ASCII CSV 
            format as <table name>.dat;
        * simulation box size in Mpc;
        * number of files in the simulation - required for the sample uploading 
            script only;
        * galaxy positions and the galaxy id columns.
    - upload the galaxy data to the database using the sample uploading script 
        bin2sql.php or bin2sql.py, or any other custom way.

To run the script from command line:

    php lightcone.php <ra min> <ra max> <dec min> <dec max> <z min> <z max> <cut> <include>

    parameters:
        * ra/dec - right ascention/declination in degrees (0..360, -90..90);
        * z - redshift of the light-cone;
        * cut - conditional expression in quotes (e.g. "StellarMass > 0.1");
        * include - additional galaxy properties to include in the catalogue
            (comma separated in quotes).

