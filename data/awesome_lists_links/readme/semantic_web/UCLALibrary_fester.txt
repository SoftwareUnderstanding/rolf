# Fester
[![Maven Build](https://github.com/uclalibrary/fester/workflows/Maven%20PR%20Build/badge.svg)](https://github.com/UCLALibrary/fester/actions) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/990b5c316e0a45d092c83d58f148e0e8)](https://www.codacy.com/gh/UCLALibrary/fester?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=UCLALibrary/fester&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/990b5c316e0a45d092c83d58f148e0e8)](https://www.codacy.com/gh/UCLALibrary/fester?utm_source=github.com&utm_medium=referral&utm_content=UCLALibrary/fester&utm_campaign=Badge_Coverage) [![Known Vulnerabilities](https://snyk.io/test/github/uclalibrary/fester/badge.svg)](https://snyk.io/test/github/uclalibrary/fester)

A microservice for facilitating the creation, storage, and retrieval of IIIF manifests and collections.

## Prerequisites

There are just a few prerequisites that must be installed, and configured correctly, in order to build Fester:

* [Java Development Kit (JDK)](https://openjdk.java.net/install/): version 11 or greater
* [Docker](https://docs.docker.com/get-docker/): version 19.03 or greater
* [Maven](https://maven.apache.org/download.cgi): version 3.6 or greater
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) [Not required, but useful]

These packages may also be available through your system's package repository. If they are, it's better to install from that source so that they will be kept up to date for you.

You will also need an account on AWS and have the ability to create [IAM](https://aws.amazon.com/iam/) accounts and [S3](https://aws.amazon.com/s3/) buckets.

## Configuring the Build

Fester uses an S3 bucket for back-end storage. To be able to run the project's tests, several configuration values must be supplied:

* fester.s3.bucket
* fester.s3.access_key
* fester.s3.secret_key
* fester.s3.region

These values can be set as properties in your system's Maven settings.xml file (or be supplied on the command line at build time).

## Building the Project

The project builds an executable Jar that can be run to start the microservice. To build the project, run:

    mvn package

This will put the executable Jar in the `target/build-artifact` directory.

To generate the site's documentation, run:

    mvn site

This will generate the documentation in the `target/site` directory.

## Configuring the Tests

The project contains unit, functional, and integration tests, with controls on how to control which tests are run. In order to run the functional and integration tests, the build machine must have a working Docker environment. Setting up Docker on your machine will depend on the type of machine you have (e.g., Linux, Mac, or Windows). Docker's [documentation](https://docs.docker.com/get-docker/) should be consulted on how to do this.

When running the build using the 'package' phase (as described above), only the unit tests are run. If you want to run all the possible tests, the project can be built with:

    mvn integration-test

or

    mvn verify

This will run the functional, feature flag, and integration tests, in addition to the unit tests. If you want to skip a particular type of test but still run the 'verify' phase, you can use one of the following arguments to your Maven command:

    -DskipUTs
    -DskipITs
    -DskipFTs
    -DskipFfTs

The first will skip the unit tests; the second will skip the integration tests; the third will skip the functional tests; and, the fourth will skip the feature flag tests. They can also be combined so that two types of tests are skipped. For instance, only the functional tests will be run if the following is typed:

    mvn verify -DskipUTs -DskipITs

When running the integration and functional tests, it may be desirable to turn on logging for the containers that run the tests. This can be useful in debugging test failures that happen within the container. To do this, supply one (or any) of the following arguments to your build:

    -DseeLogsFT
    -DseeLogsIT
    -DseeLogsFfT

This will tunnel the container's logs (including the application within the container's logs) to Maven's logging mechanism so that you will be able to see what's happening in the container as the tests are being run against it.

You might also want to adjust the logging level on the tests themselves. By default, the test loggers are configured to write DEBUG logs to a log file in the `target` directory and ERROR logs to standard out. To change the log level of the standard out logging, run Maven with the `logLevel` argument; for instance:

    mvn -DlogLevel=DEBUG test

If you want more fine-grained control over the logging, you can copy the `src/test/resources/logback-test.xml` file to the project's root directory and modify it. A `logback-test.xml` file in the project's home directory will be used instead of the standard one in `src/rest/resources` if it's available. That hypothetical file has also been added to the project's `.gitignore` so you don't need to worry about checking it into Git.

## Running a Single Test

It is sometimes useful to run a single test (instead of the whole test suite). The Surefire Maven plugin allows for this, but it's worth noting that when a single test is run in this way the test suite's pre-configured system properties are not picked up from the plugin's configuration. To work around this, a dev who wants to run a single test must supply the necessary properties theirself. For example, if one wanted to run the functional test that checks that missing images get a placeholder image in the manifest, the command to do that would be:

    mvn integration-test -Dtest=MissingImageFT -Dfester.s3.bucket=iiif-fester -Dfester.placeholder.url="https://iiif.library.ucla.edu/iiif/2/blank" -Dfester.logs.output=true

You would want to supply your own values for `fester.s3.bucket` and `fester.placeholder` of course. This command will spin up the Docker container that the functional test is run against, but it will only run the `MissingImageFT` test, skipping all the integration and other functional tests in the suite.

## Running the Application for Development

You can run a development instance of Fester by typing the following within the project root:

    mvn -Plive test

Once run, the service can be verified/accessed at [http://localhost:8888/fester/status](http://localhost:8888/fester/status). The API documentation can be accessed at [http://localhost:8888/fester/docs](http://localhost:8888/fester/docs)

## Debugging with Eclipse IDE

There are two ways to debug Fester:

- **Debugging the tests.** This enables the developer to step through both the test and application code as the test suite runs.
- **Debugging a running instance.** This enables the developer to step through the application code as they interact with the HTTP API.

The following setup instructions were tested with [Eclipse IDE](https://www.eclipse.org/eclipseide/) 4.14.0 (2019-12).

### Debugging the tests

From within Eclipse:

1. Create a new debug configuration
    - In the top-level menu, select *Run* > *Debug Configurations...*
    - In the pop-up window:
        - Create a new configuration of type *Remote Java Application*
        - Set *Name* to something like `Fester (JDWP server for containerized instances created by test suite)`
        - In the *Connect* tab:
            - Set *Project* to the Fester project directory
            - Set *Connection Type* to `Standard (Socket Listen)`
            - Set *Port* to `5556`
            - Set *Connection limit* to `16`
            - Check *Allow termination of remote VM* (optional)
2. Create another debug configuration *
    - In the top-level menu, select *Run* > *Debug Configurations...*
    - In the pop-up window:
        - Create a new configuration of type *Maven Build*
        - Set *Name* to something like `Fester (debug test suite)`
        - In the *Main* tab:
            - Set *Base directory* to the Fester project directory
            - Set *Goals* to `integration-test`
            - Set *Profiles* to `debug`
            - Set *User settings* to the path to a `settings.xml` that contains your AWS S3 credentials
3. Run the debug configuration created in Step 1 **
4. Run the debug configuration created in Step 2 **

_* As an alternative to step 2 (and 4), run the following from the command line (after completing steps 1 and 3):_

    mvn -Pdebug integration-test

_** If you're doing this for the first time, you may need to bring back the pop-up window where you created the configuration in order to invoke it. Otherwise, you can use toolbar buttons, or hotkeys <kbd>Ctrl</kbd> <kbd>F11</kbd> (Run) or <kbd>F11</kbd> (Debug)._

### Debugging a running instance

This procedure will start an instance of Fester with port `5555` open for incoming JDWP connections.

From within Eclipse:

1. Create a new run configuration ***
    - In the top-level menu, select *Run* > *Run Configurations...*
    - In the pop-up window:
        - Create a new configuration of type *Maven Build*
        - Set *Name* to something like `Fester (debugging mode)`
        - In the *Main* tab:
            - Set *Base directory* to the Fester project directory
            - Set *Goals* to `test`
            - Set *Profiles* to `runDebug`
            - Set *User settings* to the path to a `settings.xml` that contains your AWS S3 credentials
2. Create a new debug configuration
    - In the top-level menu, select *Run* > *Debug Configurations...*
    - In the pop-up window:
        - Create a new configuration of type *Remote Java Application*
        - Set *Name* to something like `Fester (JDWP client)`
        - In the *Connect* tab:
            - Set *Project* to the Fester project directory
            - Set *Connection Type* to `Standard (Socket Attach)`
            - Set *Host* to `localhost`
            - Set *Port* to `5555`
            - Check *Allow termination of remote VM* (optional)
3. Run the new run configuration created in Step 1
4. Run the new debug configuration created in Step 2

_*** As an alternative to step 1 (and 3), run the following from the command line:_

    mvn -PrunDebug test

_and then proceed with steps 2 and 4._

## Load Testing

A [Locust](https://docs.locust.io/en/stable/index.html) test file is included, it only tests PUTs of manifests. If you wish to run the test, you need to have Locust installed, and then run the following command from the src/test/scripts/locust folder:

    locust --host=url-of-the-server-you-are-testing

For example, if you wish to run a Locust test against a dev instance on your own machine, you would enter:

    locust --host=http://localhost:8888

## Git Hooks

To prevent accidentally pushing commits that would cause the CI build to fail, you can configure your Git client to use a pre-push hook:

    ln -s ../../src/test/scripts/git-hooks/pre-push .git/hooks

## Working with Pinned OS Packages

We pin the versions of packages that we install into our base image. What this means is that periodically a pinned version will become obsolete and the build will break. We have a nightly build that should catch this issues for us, but in the case that you find the breakage before us, there is a handy way to tell which pinned version has broken the build. To see the current versions inside the base image, run:

    mvn validate -Dversions

This will output a list of current versions, which can be compared to the pinned versions defined in the project's POM file (i.e., pom.xml).

## Festerize

[Festerize](https://github.com/UCLALibrary/festerize) may be used to interact with Fester, as an alternative to the built-in CSV upload form.

## Contact

We use an internal ticketing system, but we've left the GitHub [issues](https://github.com/UCLALibrary/fester/issues) open in case you'd like to file a ticket or make a suggestion. You can also contact Kevin S. Clarke at <a href="mailto:ksclarke@ksclarke.io">ksclarke@ksclarke.io</a> if you have a question about the project.
