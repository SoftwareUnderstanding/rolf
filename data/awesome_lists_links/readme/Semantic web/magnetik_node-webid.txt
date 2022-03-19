#node-webid

Node.js module with tools to help using WebID (http://www.webid.info).

##Installation

### [npm](https://npmjs.org/package/webid)

Just require the module `webid`:

```
var webid = require('webid');
```

### Manual

Start `cake build` and get the webid.js in the bin folder.

## Usage

Check the project [webid-demo](https://github.com/magnetik/node-webid-demo) to see a working example. 

Basic usage:

```
var webid = require('webid');
var verifAgent = new webid.VerificationAgent(certificate);
	verifAgent.verify(function (result) {
		//Success! User is identified
		var foaf = new webid.Foaf(result);
		req.session.profile = foaf.parse();
	}, function(result) {
		//An error occured
	});
```

##Licence

The lib is available under MIT Licence: http://www.opensource.org/licenses/MIT

