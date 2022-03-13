csv2rdf
=======

Ruth Helfinstein
6/24/2012

CSV to RDF translator
Does a simple CSV to RDF translation.  
- Assumes the first line contains the attribute names
- Assumes all instances are separated by newlines and contain the right number of elements
- Items in the csv file may have quotes or not, it works either way.


Outputs the rdf file as <filename>.rdf
NOTE: It does not currently check if the output file already exists and will overwrite it if it does.

Looks for a configuration file called <filename>-config.csv, where <filename>.csv is the input file.
Each line in the configuration file describes one item in the header line of the input csv file
<csv_name>, <type>(<optional rdf_name>), <class>

where <csv_name> is the name seen in the header of the column in the csv file. 
<type> is either "class" or "property" or "ignore" (without quotes)
<rdf_name> in parenthesis is optional, indicates name to use for this property or class in the rdf file
<class> is the class (for a property) or superclass (for a class) 

the lines in the config file do not have to be in the same order as the columns in the csv file
with the exception of any columns with a blank name.  These will be renamed "unlabeled1" "unlabeled2" 
and will only match if they are in the same order.

NOTE: any columns NOT listed in the configuration file will be ignored in the rdf file.

If no configuration file is found, the program will create a basic one

Files:
csv2rdf.jar
example.csv  - sample input file
example-config.csv - sample config file
src - source files

To run CSV to RDF from the console:
java -jar csv2rdf.jar <input csv file name>
*** NOTE You can specify the name of the .csv file to use on the command line, otherwise it will default to input.csv



for example 
	java -jar csv2rdf.jar input.csv
creates
	input.rdf

Changes 2012.08.02
• Add configuration file

Changes 2012.07.04
• Ignore leading or trailing spaces in the csv
• Add a unlabeled1,unlabled2, unlabeld3... to any attribute in the csv that is not named.
• If there is a blank in the data, do not  add the property name to instance  
• Add additional input question: Do you want an 'all' class? 
If the user answers yes an "all" class is created as the
only subclass to Thing and everything  is subclassed to "all".




