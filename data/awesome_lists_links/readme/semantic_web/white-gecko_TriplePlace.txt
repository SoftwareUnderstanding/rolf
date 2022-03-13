TriplePlace
===========

This is TriplePlace a light weight and flexible Triple Store for Android. It uses a indexing structure similar to the
one in [Hexastore](http://www.zora.uzh.ch/8938/2/hexastore.pdf). TriplePlace uses
[TokyoCabinet](http://fallabs.com/tokyocabinet/) as persistent storage system. I've also published a patched version of [TokyoCabinet](https://github.com/white-gecko/TokyoCabinet) and TokyoCabinet-Java-API resp. [TokyoCabinet-Android-API](https://github.com/white-gecko/TokyoCabinet-Android-API).

TriplePlace is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

TriplePlace is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with TriplePlace.
If not, see <[http://www.gnu.org/licenses/](http://www.gnu.org/licenses/)>.

To be done/To be implemented
----------

- Implement some kind of logging to implement atomicity and consistency (ACID) between the indices
- Implement edit and maybe delete operations (norman says delete would be to expencive, maybe we have to mark those
triples as deleted)
- Implement query on graph patherns
- Implement RDF/XML, N-Triples/Turtle/N3 and ... import/export
