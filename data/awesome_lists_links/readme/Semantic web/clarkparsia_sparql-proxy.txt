## Simple SPARQL Endpoint Proxy Servlet

Sparql Endpoint Proxy Servlet - Helps for bypassing CORS issues.

Simple proxy that redirects all elements in the request to a configured SPARQL endpoint. Headers in the request are also redirected.

### Configuring the proxy

The only setting required to configure the SPARQL Endpoint proxy is `proxy.host` which refers to the host:port info in which the Stardog HTTP service is running, for example:

    <init-param>
		<param-name>proxy.host</param-name>
		<param-value>http://localhost:5822</param-value>
	</init-param>

The previous indicats the a Stardog DB is running the HTTP protocol in `http://localhost:5822`, which can be modified to any other information, either a local reference or remote.

### Building .war file with Ant

To build the proxy war, just execute:

    ant clean web-dist
    
This will generate the `sparql-proxy.war` file that you just need to copy to your app container `webapps` directory.

### Using the proxy

To use the proxy, simple send the request to it, following the same pattern as you would do with a Stardog DB directly. For instance, to query the DB `gov` directly to Stardog HTTP endpoint you'll do:

    curl -X GET "http://localhost:5822/gov/query?query=select%20*%20where%20%7B%20%3Fs%20%3Fp%20%3Fo%20%7D%20limit%2010"
    
Using the proxy (with CORS support), the same can be done:

    curl -X GET "http://localhost:8181/sparql-proxy/gov/query?query=select%20*%20where%20%7B%20%3Fs%20%3Fp%20%3Fo%20%7D%20limit%2010"
    
where:

* `http://localhost:8181/sparql-proxy` is the path for the sparql-proxy (with servlet containter running in port 8181)
* `gov` is the Stardog DB
* `query?query=` points to the encoded query.

