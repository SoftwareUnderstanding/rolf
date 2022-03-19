Sparks JavaScript Library
==================

Sparks is a set of JavaScript libraries designed for simplifying the access to RDF data.
Sparks is licensed under the MIT license.

Disclaimer
---------------
This is the first release of the library and it only contains an initial version of the Prism filtering framework.
There is some known bugs and the library doesn't support the upcoming Sparks Plug library.


Install
--------
Sparks depends on [JQuery](http://jquery.com/) and [JavascriptMVC](http://javascriptmvc.com/) in order to work properly.
Download [JavascriptMVC](http://javascriptmvc.com/)  and put the Sparks code in sparks directory inside [JavascriptMVC](http://javascriptmvc.com/).

Alternatively, you might use the file ''*sparks.prism.js*'' which includes everything except [JQuery](http://jquery.com/).

Usage
---------
```javascript
//You need to include the sparks library before doing the following:
//- Create a new prism endpoint:
var prism = new Sparks.Prism("http://nebula.dcs.shef.ac.uk/sparks/sparql","?root a <http://ext.dcs.shef.ac.uk/~u0080/linkedPOI/core#POI>");

//- Instantiate some lenses:
$('#tags-list').sparks_prism_lens_tags({prism: prism});
$('#search').sparks_prism_lens_search({prism: prism});
$('#list').sparks_prism_lens_list({prism: prism});
```
