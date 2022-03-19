Vert.x Blazegraph Service
=========================

This service provides an asynchronous interface around Blazegraph.

Before use this service, place it into Maven local repo using this task:

```
$ gradle publishToMavenLocal
```

If you want to use this service from Maven local, use this in your dependencies part of your `build.gradle` file:

```
dependencies {
    ...
    ...
    compile 'name.bpdp.vertx:blazegraph-service:1.0.0'
    ...
    ...
}
```
