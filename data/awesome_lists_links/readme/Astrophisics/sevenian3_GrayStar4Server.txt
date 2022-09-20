# GrayStar4Server

To see Web app: http://www.ap.smu.ca/~ishort/OpenStars/GrayStarServer/grayStarServer.html

Web-based pedagogical stellar astrophysics
Stellar atmospheric modeling and atomic-line-list-driven spectrum synthesis on the server side in Java.
Client-side JavaScript code produces synthetic observables and measurments, applies additional
spectral line broadening, and renders direct observables and draws technical plots in HTML
with <canvas> element.
Synthetic spectrum is labeled with line-ID tags and the HTML UI allows the user to inspect
the ascii version of the atomic line list (from NIST).
JSON used to mark-up atmospheric modeling and spectrum synthesis output for transmission with
http rquest.
Clinet-side JS code contains version of IDL's axis() procedure for automatically scaling
and graduating x- and y-axis data.  
Open source MIT license.  Computatational astrophysicists: This is also a path-finder for 
webifying research-grade modeling that would bnefit from citizen science job-control level parallelism. 
