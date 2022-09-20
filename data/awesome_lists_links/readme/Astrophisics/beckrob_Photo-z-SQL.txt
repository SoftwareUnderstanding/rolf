# Photo-z-SQL

Photo-z-SQL is a database-integrated, template-based photometric redshift estimation software package, written in C\#. Details are presented in the following paper: [https://arxiv.org/abs/1611.01560].

Comments, questions and requests about features/bugs/etc. should be sent to the lead author, Robert Beck.

## Installation instructions

First, it is highly recommended to install the most recent version of SQLArray to your database server, from https://github.com/idies/sqlarray. Debug and Release packages are bundled in the Install directory, alongside a pdf with instructions.

While the required assemblies are also included in the Photo-z-SQL package, a separate SQLArray DB installation is needed to have access to many SQLArray SQL functions that may help with setting up and manipulating the variable-length arrays.

The installation process of Photo-z-SQL is very similar to that of SQLArray.

1. On the SQL server instance *servername\\instance*, choose or create a new database with name *databasename* that will store the Photo-z-SQL functions. (E.g. we use *databasename* = PhotoZSQL.)

2. CLR integration has to enabled in this new database using the following commands:

	```
	EXEC sp_configure 'show advanced options' , '1';
	go
	reconfigure;
	go
	EXEC sp_configure 'clr enabled' , '1'
	go
	reconfigure;
	-- Turn advanced options back off
	EXEC sp_configure 'show advanced options' , '0';
	go
	```

3. Since the Photo-z-SQL assemblies require unrestricted access, the database will need to be set to trustworthy using the commands below. (Alternatively, an RSA key could be used, see the SQLArray documentation.)
	
	<pre><code>ALTER DATABASE <i>databasename</i>
	SET TRUSTWORTHY ON
	GO
	</code></pre>

4. Now that the database has been prepared, Photo-z-SQL can be installed and removed using the provided scripts. Navigate to Jhu.PhotoZSQL\bin\Release (or Jhu.PhotoZSQL\bin\Debug for the debug version), and execute the Create (install) script:

	<pre><code>sqlcmd -S <i>servername\instance</i> -E -d <i>databasename</i> -i Jhu.PhotoZSQLDB.Create.sql
	</code></pre>
	
	The `-E` flag denotes integrated Windows authentication, for SQL Server authentication `-U `*`username`*`-P `*`password`* can be used instead.

5. The packages and assemblies can be uninstalled similarly with the Drop (uninstall) script:

	<pre><code>sqlcmd -S <i>servername\instance</i> -E -d <i>databasename</i> -i Jhu.PhotoZSQLDB.Drop.sql
	</code></pre>
	
6. Now the Photo-z-SQL functions can be called through the database where they were installed. Note that since Compute is a SQL keyword, square brackets are needed to resolve the schema in the case of Compute functions.

	<pre><code>SELECT <i>databasename</i>.Config.RemoveInitialization()
	...
	SELECT <i>databasename</i>.[Compute].PhotoZMinChiSqr_ID(...)
	</code></pre>

## Function description

### Config functions

* Config.AddMissingValueSpecifier - Specifies a value that denotes missing or otherwise invalid inputs. All such inputs are ignored.
	* Float, @aMissingValue - the value that should be ignored.
	* Returns int, 0 (should not fail under normal circumstances).

* Config.ClearMissingValueSpecifiers - Removes all previously set missing value specifiers. All further inputs will be processed.
	* No parameters.
	* Returns int, 0 (should not fail under normal circumstances).

* Config.RemoveExtinctionLaw - Removes the data associated with the extinction law -- Galactic extinction correction is no longer applied.
	* No parameters.
	* Returns int, 0 (should not fail under normal circumstances).
	
* Config.RemoveInitialization - Removes all data connected to the photo-z configuration, including cached filters, templates and synthetic magnitudes.
	* No parameters.
	* Returns int, 0 (should not fail under normal circumstances).
	
* Config.SetupAbsoluteMagnitudeLimitPrior_ID - Setup a prior with a cut at a given minimum absolute magnitude (spectra brighter than this are excluded), in a reference filter specified with a VO Filter Profile Services ID. Note that the distance modulus calculation will diverge at redshift 0, so use a suitably small number (e.g. 0.001) as the start of the coverage. 
	* Int, @referenceFilterID - the integer ID of the reference filter (e.g. 16 for the SDSS r-band).
	* Float, @absMagLimit - the absolute magnitude specifying the cut.
	* Float, @h - the normalized Hubble parameter, e.g. 0.7.
	* Float, @Omega_m - the given cosmological parameter, e.g. 0.3.
	* Float, @Omega_lambda - the given cosmological parameter, e.g. 0.7.
	* Returns int, 0 if successful, or a negative error code.
	
* Config.SetupAbsoluteMagnitudeLimitPrior_URL - The same as above, with the reference filter specified via an URL. Non-VO filters can be used, provided that the VO format is followed.
	* Nvarchar(max), @referenceFilterURL - the URL address of the reference filter as a string (e.g. 'http://voservices.net/filter/filterascii.aspx?FilterID=16' for the SDSS r-band).
	* Float, @absMagLimit - the absolute magnitude specifying the cut.
	* Float, @h - the normalized Hubble parameter, e.g. 0.7.
	* Float, @Omega_m - the given cosmological parameter, e.g. 0.3.
	* Float, @Omega_lambda - the given cosmological parameter, e.g. 0.7.
	* Returns int, 0 if successful, or a negative error code.
	
* Config.SetupBenitezHDFPrior_ID - Setup the Benitez (2000) HDF-N prior, using the WFPC2 F814W filter specified with a VO Filter Profile Services ID. Note that the template list has to be already set up when calling this, since the prior has to be adapted to the templates.
	* Int, @referenceFilterID - the integer ID of the WFPC2 F814W reference filter (54 in the VO list).
	* Bit, @usingLePhareTemplates - boolean, 0 if the 71 BPZ templates are used, 1 if the 641 Le Phare templates are used.
	* Returns int, 0 if successful, or a negative error code.
	
* Config.SetupBenitezHDFPrior_URL - The same as above, with the reference filter specified via an URL.
	* Nvarchar(max), @referenceFilterURL - the URL address of the WFPC2 F814W reference filter as a string ('http://voservices.net/filter/filterascii.aspx?FilterID=54')
	* Bit, @usingLePhareTemplates - boolean, 0 if the 71 BPZ templates are used, 1 if the 641 Le Phare templates are used.
	* Returns int, 0 if successful, or a negative error code.
	
* Config.SetupExtinctionLaw_ID - Setup the extinction law to use when correcting for Galactic extinction. The reference spectrum used in the computations is specified via a VO Spectrum Services ID.
	* Int, @referenceSpectrumID - the integer ID of the reference (usually elliptical) spectrum (e.g. 511 in the VO list).
	* Float, @dustParameterR_V - the R_V dust parameter to be used (e.g. 3.1).
	* Bit, @useFitzpatrickExtinctionInsteadOfODonnell - boolean, 1 if the Fitzpatrick (1999) extinction law should be used, 0 if the combination of O'Donnell (1994) and Cardelli (1989) extinction laws should be used.
	* Returns int, 0 if successful, or a negative error code.
	
* Config.SetupExtinctionLaw_URL - The same as above, with the reference spectrum specified via an URL. Non-VO spectra can be used, provided that the VO format is followed.
	* Nvarchar(max), @referenceSpectrumURL - the URL address of the reference (usually elliptical) spectrum as a string (e.g. 'http://voservices.net/spectrum/search_details.aspx?format=ascii&id=ivo%3a%2f%2fjhu%2ftemplates%23511').
	* Float, @dustParameterR_V - the R_V dust parameter to be used (e.g. 3.1).
	* Bit, @useFitzpatrickExtinctionInsteadOfODonnell - boolean, 1 if the Fitzpatrick (1999) extinction law should be used, 0 if the combination of O'Donnell (1994) and Cardelli (1989) extinction laws should be used.
	* Returns int, 0 if successful, or a negative error code.
	
* Config.SetupFlatPrior - Setup a flat prior, with all combinations of parameters having equal prior probability (this is the default choice).
	* No parameters.
	* Returns int, 0 if successful, or a negative error code.

* Config.SetupTemplateList_ID - Specifies the list of SED templates to use in the photo-z estimation, along with the resolution in redshift and luminosity. The templates are specified with VO Spectrum Services IDs.
	* Varbinary(max), @templateIDList - a list of VO integer IDs in a SQLArray IntArrayMax object (440 to 510 are the BPZ templates, 511 to 1151 are the Le Phare templates).
	* Float, @fluxMultiplier - a flux multiplier can be used to calibrate the flux of the templates, here this is not needed and can be set to 1.0.
	* Float, @redshiftFrom - the start of the redshift coverage.
	* Float, @redshiftTo - the end of the redshift coverage.
	* Float, @redshiftStep - the step size/multiplier between subsequent redshift values of the coverage.
	* Bit, @logarithmicRedshiftSteps - boolean, 0 if linear steps of size @redshiftStep should be taken, 1 if the redshift coverage should be logarithmic, with @redshiftStep as the multiplier.
	* Int, @luminosityStepNumber - the integer number of steps that should be taken around the best-fitting luminosity, chosen to cover a 3-sigma magnitude range. Has to be an odd number (center, plus two sides), generally 11 is more than enough, can be reduced all the way down to 1 to increase computational speed.
	* Returns int, the number of template spectra successfully read, or a negative error code.
	
* Config.SetupTemplateList_URL - The same as above, with the templates specified with URLs. Non-VO spectra can be used, provided that the VO format is followed.
	* Nvarchar(max), @templateURLList - a list of spectrum URLs in a string, separated by the character '|'.
	* Float, @fluxMultiplier - a flux multiplier can be used to calibrate the flux of the templates, here this is not needed and can be set to 1.0.
	* Float, @redshiftFrom - the start of the redshift coverage.
	* Float, @redshiftTo - the end of the redshift coverage.
	* Float, @redshiftStep - the step size/multiplier between subsequent redshift values of the coverage.
	* Bit, @logarithmicRedshiftSteps - boolean, 0 if linear steps of size @redshiftStep should be taken, 1 if the redshift coverage should be logarithmic, with @redshiftStep as the multiplier.
	* Int, @luminosityStepNumber - the integer number of steps that should be taken around the best-fitting luminosity, chosen to cover a 3-sigma magnitude range. Has to be an odd number (center, plus two sides), generally 11 is more than enough, can be reduced all the way down to 1 to increase computational speed.
	* Returns int, the number of template spectra successfully read, or a negative error code.
	
* Config.SetupTemplateList_LuminositySpecified_ID - The same as SetupTemplateList_ID, but the luminosity range to cover is explicitly given. The fluxes of the templates generally need to be scaled here.
	* Varbinary(max), @templateIDList - a list of VO integer IDs in a SQLArray IntArrayMax object (440 to 510 are the BPZ templates, 511 to 1151 are the Le Phare templates).
	* Float, @fluxMultiplier - a flux multiplier, needed to calibrate the flux of the templates to the adopted luminosity range. The scaled templates are then assumed to have unit luminosity.
	* Float, @redshiftFrom - the start of the redshift coverage.
	* Float, @redshiftTo - the end of the redshift coverage.
	* Float, @redshiftStep - the step size/multiplier between subsequent redshift values of the coverage.
	* Bit, @logarithmicRedshiftSteps - boolean, 0 if linear steps of size @redshiftStep should be taken, 1 if the redshift coverage should be logarithmic, with @redshiftStep as the multiplier.
	* Float, @luminosityFrom - the start of the luminosity coverage.
	* Float, @luminosityTo - the end of the luminosity coverage.
	* Float, @luminosityStep - the step size/multiplier between subsequent luminosity values of the coverage.
	* Bit, @logarithmicLuminositySteps - boolean, 0 if linear steps of size @luminosityStep should be taken, 1 if the luminosity coverage should be logarithmic, with @luminosityStep as the multiplier.
	* Returns int, the number of template spectra successfully read, or a negative error code.
	
* Config.SetupTemplateList_LuminositySpecified_URL - The same as SetupTemplateList_URL, but the luminosity range to cover is explicitly given. The fluxes of the templates generally need to be scaled here.
	* Nvarchar(max), @templateURLList - a list of spectrum URLs, separated by the character '|'.
	* Float, @fluxMultiplier - a flux multiplier, needed to calibrate the flux of the templates to the adopted luminosity range. The scaled templates are then assumed to have unit luminosity.
	* Float, @redshiftFrom - the start of the redshift coverage.
	* Float, @redshiftTo - the end of the redshift coverage.
	* Float, @redshiftStep - the step size/multiplier between subsequent redshift values of the coverage.
	* Bit, @logarithmicRedshiftSteps - boolean, 0 if linear steps of size @redshiftStep should be taken, 1 if the redshift coverage should be logarithmic, with @redshiftStep as the multiplier.
	* Float, @luminosityFrom - the start of the luminosity coverage.
	* Float, @luminosityTo - the end of the luminosity coverage.
	* Float, @luminosityStep - the step size/multiplier between subsequent luminosity values of the coverage.
	* Bit, @logarithmicLuminositySteps - boolean, 0 if linear steps of size @luminosityStep should be taken, 1 if the luminosity coverage should be logarithmic, with @luminosityStep as the multiplier.
	* Returns int, the number of template spectra successfully read, or a negative error code.
	
* Config.SetupTemplateTypePrior - Sets up a template type prior, with a list of prior probabilities corresponding to every template spectrum. The probabilities should be given in the same order as in the case of the @templateURLList or @templateIDList parameter of the SetupTemplateList function call.
	* Varbinary(max), @templateProbabilities - a list of float probabilities in a SQLArray FloatArrayMax object.
	* Returns int, 0 if successful, or a negative error code.

### Compute functions

* [Compute].PhotoZMinChiSqr_ID - Perform a minimum chi-square photo-z fit to the given flux or magnitude data. Corresponding photometric filters are specified with VO Filter Profile Services IDs.
	* Varbinary(max), @magOrFluxArray - a list of fluxes or magnitudes in a SQLArray FloatArrayMax object.
	* Varbinary(max), @magOrFluxErrorArray - a list of flux or magnitude errors in a SQLArray FloatArrayMax object, in the same order as in @magOrFluxArray.
	* Bit, @inputInMags - boolean, 0 if the input arrays contain fluxes, 1 if they contain AB magnitudes.
	* Varbinary(max), @filterIDList - a list of filter integer IDs in a SQLArray IntArrayMax object, in the same order as in @magOrFluxArray.
	* Float, @extinctionMapValue - the Schlegel (1998) dust map value to be used for extinction correction. If 0.0 or the extinction law has not been set up, no correction is applied.
	* Bit, @fitInFluxSpace - boolean, 0 if the fitting should be performed in magnitude space, 1 if it should be done in flux space. Conversions are performed as needed.
	* Float, @errorSmoothening - the independent error term to be added, in magnitudes.
	* Returns float, the best-fitting redshift value, or a negative error code.

* [Compute].PhotoZMinChiSqr_URL - The same as above, but filters are specified with URLs. Non-VO filters can be used, provided that the VO format is followed.
	* Varbinary(max), @magOrFluxArray - a list of fluxes or magnitudes in a SQLArray FloatArrayMax object.
	* Varbinary(max), @magOrFluxErrorArray - a list of flux or magnitude errors in a SQLArray FloatArrayMax object, in the same order as in @magOrFluxArray.
	* Bit, @inputInMags - boolean, 0 if the input arrays contain fluxes, 1 if they contain AB magnitudes.
	* Nvarchar(max), @filterURLList - a list of filter URLs in a string, separated by the character '|', in the same order as in @magOrFluxArray.
	* Float, @extinctionMapValue - the Schlegel (1998) dust map value to be used for extinction correction. If 0.0 or the extinction law has not been set up, no correction is applied.
	* Bit, @fitInFluxSpace - boolean, 0 if the fitting should be performed in magnitude space, 1 if it should be done in flux space. Conversions are performed as needed.
	* Float, @errorSmoothening - the independent error term to be added, in magnitudes.
	* Returns float, the best-fitting redshift value, or a negative error code.

* [Compute].PhotoZBayesian_ID - Perform Bayesian photo-z estimation using the given flux or magnitude data. Corresponding photometric filters are specified with VO Filter Profile Services IDs.
	* Varbinary(max), @magOrFluxArray - a list of fluxes or magnitudes in a SQLArray FloatArrayMax object.
	* Varbinary(max), @magOrFluxErrorArray - a list of flux or magnitude errors in a SQLArray FloatArrayMax object, in the same order as in @magOrFluxArray.
	* Bit, @inputInMags - boolean, 0 if the input arrays contain fluxes, 1 if they contain AB magnitudes.
	* Varbinary(max), @filterIDList - a list of filter integer IDs in a SQLArray IntArrayMax object, in the same order as in @magOrFluxArray.
	* Float, @extinctionMapValue - the Schlegel (1998) dust map value to be used for extinction correction. If 0.0 or the extinction law has not been set up, no correction is applied.
	* Bit, @fitInFluxSpace - boolean, 0 if the fitting should be performed in magnitude space, 1 if it should be done in flux space. Conversions are performed as needed.
	* Float, @errorSmoothening - the independent error term to be added, in magnitudes.
	* Returns a table with two columns, Redshift (float), a given point of the redshift grid, and Probability (float), the corresponding normalized probability. Upon failure, a (negative error code, 0) row is returned.

* [Compute].PhotoZBayesian_URL - The same as above, but filters are specified with URLs. Non-VO filters can be used, provided that the VO format is followed.
	* Varbinary(max), @magOrFluxArray - a list of fluxes or magnitudes in a SQLArray FloatArrayMax object.
	* Varbinary(max), @magOrFluxErrorArray - a list of flux or magnitude errors in a SQLArray FloatArrayMax object, in the same order as in @magOrFluxArray.
	* Bit, @inputInMags - boolean, 0 if the input arrays contain fluxes, 1 if they contain AB magnitudes.
	* Nvarchar(max), @filterURLList - a list of filter URLs in a string, separated by the character '|', in the same order as in @magOrFluxArray.
	* Float, @extinctionMapValue - the Schlegel (1998) dust map value to be used for extinction correction. If 0.0 or the extinction law has not been set up, no correction is applied.
	* Bit, @fitInFluxSpace - boolean, 0 if the fitting should be performed in magnitude space, 1 if it should be done in flux space. Conversions are performed as needed.
	* Float, @errorSmoothening - the independent error term to be added, in magnitudes.
	* Returns a table with two columns, Redshift (float), a given point of the redshift grid, and Probability (float), the corresponding normalized probability. Upon failure, a (negative error code, 0) row is returned.

	
### Util functions	
	
* Util.GetTotalCLRMemoryUsage - Gets the total amount of currently allocated CLR memory in the server.
	* No parameters.
	* Returns bigint, total memory used in bytes.
	
* Util.ParseDoubleArray1D - Efficiently parses a string into a one-dimensional SQLArray FloatArrayMax object. (Much faster than the more general SqlArray.FloatArrayMax.Parse function.)
	* Nvarchar(max), @doubleListAsStr - the string containing the floats, with a format following '[1.0,2.1,3.2]', but with the list separator from the server's current culture setting instead of the comma.
	* Returns varbinary(max), the parsed SQLArray FloatArrayMax object.
	
* Util.ParseDoubleArray1DInvariant - Efficiently parses a string into a one-dimensional SQLArray FloatArrayMax object. (Much faster than the more general SqlArray.FloatArrayMax.ParseInvariant function.)
	* Nvarchar(max), @doubleListAsStr - the string containing the floats, with a format following '[1.0,2.1,3.2]', using invariant culture settings.
	* Returns varbinary(max), the parsed SQLArray FloatArrayMax object.	
	
* Util.ParseIntArray1D - Efficiently parses a string into a one-dimensional SQLArray IntArrayMax object. (Much faster than the more general SqlArray.IntArrayMax.Parse function.)
	* Nvarchar(max), @intListAsStr - the string containing the integers, with a format following '[1,2,3]', but with the list separator from the server's current culture setting instead of the comma.
	* Returns varbinary(max), the parsed SQLArray IntArrayMax object.	
	
* Util.ParseIntArray1DInvariant - Efficiently parses a string into a one-dimensional SQLArray IntArrayMax object. (Much faster than the more general SqlArray.IntArrayMax.ParseInvariant function.)
	* Nvarchar(max), @intListAsStr - the string containing the integers, with a format following '[1,2,3]', using invariant culture settings.
	* Returns varbinary(max), the parsed SQLArray IntArrayMax object.		
	
## SQLArray note

SQLArray contains built-in functions for efficiently creating 1D arrays between length *N*=1 and *N*=10 from numerical variables (e.g. SqlArray.FloatArrayMax.Vector_*N*(...) or SqlArray.IntArrayMax.Vector_*N*(...)). However, for arrays longer than that, the Parse functions are needed, which are rather general, but highly inefficient in the case of 1D arrays.

For this reason, when more than 10 elements are needed in an array, Photo-z-SQL users are encouraged to use the Util functions we provide for interfacing between Photo-z-SQL and SQLArray.

## Acknowledgements

The realization of this work was supported by the Hungarian NKFI NN grant 114560. Robert Beck was supported through the New National Excellence Program of the Ministry of Human Capacities, Hungary.
