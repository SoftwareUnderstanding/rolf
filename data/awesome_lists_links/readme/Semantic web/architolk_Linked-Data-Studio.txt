# Linked Data Studio
The Linked Data Studio (LDS) is a platform for the creation of Linked Data.

The LDS is an extension to the [Linked Data Theatre](https://github.com/architolk/Linked-Data-Theatre), and you should have a working version of the Linked Data Theatre if you want to use the Linked Data Theatre!

See [BUILD.md](BUILD.md) for instructions to build the Linked Data Studio. You can also try one of the releases:

- [lds-1.7.0.war](https://github.com/architolk/Linked-Data-Studio/releases/download/v1.7.0/lds-1.7.0.war "ldt-1.7.0.war")

If you want to create a new release of the LDS, please look into the instructions for creating a new release of the [Linked Data Theatre](https://github.com/architolk/Linked-Data-Theatre), the instructions are the same.

To deploy the Linked Data Studio in a tomcat container, follow the instructions in [DEPLOY.md](DEPLOY.md). You can also opt for a docker installation, see [DOCKER.md](DOCKER.md).

The Linked Data Studio uses a configuration graph containing all the triples that make up the LDS configuration. Instructions and examples how to create such a configuration can be found at the [wiki](https://github.com/architolk/Linked-Data-Theatre/wiki) of the Linked Data Theatre.

For security, firewall, proxy and certificates: use the documentation of the [Linked Data Theatre](https://github.com/architolk/Linked-Data-Theatre).
