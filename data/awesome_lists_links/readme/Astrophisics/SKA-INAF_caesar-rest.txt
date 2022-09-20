<p align="left">
  <img src="share/CAESAR-REST_Architecture.png" alt="Software architecture"/ width="500" height="500">
</p>

# caesar-rest
caesar-rest is a REST-ful web service for astronomical source extraction and classification with the caesar source extractor [https://github.com/SKA-INAF/caesar]. The software is developed in python and consists of a few containerized microservices, deployable on standalone servers or on a distributed cloud infrastructure. The core component is the REST web application, based on the Flask framework and running behind a nginx+uwsgi http server, and providing APIs for managing the input data (e.g. data upload/download/removal) and source finding jobs (e.g. submit, get status, get outputs) with different job management systems (Kubernetes, Slurm, Celery). Additional services (AAI, user DB, log storage, job monitor, accounting) enable the user authentication, the storage and retrieval of user data and job information, the monitoring of submitted jobs, and the aggregation of service logs and user data/job stats.
Besides caesar, we also foresee to integrate other tools widely used in the radio community (e.g. Aegean, PyBDSF) and newly developed source finders based on deep learning models.    

## **Status**
This software is under development. Tested originally with python 2.7 but switched to python 3.6 later on (some apps are only available for python 3). 

## **Credit**
This software is distributed with GPLv3 license. If you use caesar-rest for your research, please add repository link or acknowledge authors in your papers.   

## **Installation**  

### **Install dependencies**
To run caesar rest service you need to install the following tools:  

* Flask [https://palletsprojects.com/p/flask/]     
* uwsgi [https://uwsgi-docs.readthedocs.io/en/latest/index.html]   
* nginx [https://nginx.org/]   
* mongodb [https://www.mongodb.com/]   
* flask-pymongo python module [https://flask-pymongo.readthedocs.io/en/latest/]  
* structlog python module [https://www.structlog.org/en/stable/]     

For the Celery-based job management, you need to install celery, a broker and a result backend service:   

* celery [http://www.celeryproject.org/]   
* broker: rabbitmq [https://www.rabbitmq.com/]    
* result backend: redis [https://redis.io/] or mongodb [https://www.mongodb.com/]     

For the Kubernetes-based job management, you need to install the Kubernetes python client library:    

* kubernetes [https://pypi.org/project/kubernetes/]

For the Slurm-based job management, you need to install these python modules:    

* requests [https://docs.python-requests.org/en/master/]    
* jwt [https://pypi.org/project/jwt/]     

To enable OpenID Connect based authentication you need to install:    

* flask-oidc-ex python module [https://pypi.org/project/flask-oidc-ex/] 

To enable log forwarding to a LogStash/ElasticSearch service, you need to install the filebeat service:    

* filebeat [https://www.elastic.co/guide/en/beats/filebeat/index.html]    

### **Package installation**
To build and install the package:    

* Create a local install directory, e.g. ```$INSTALL_DIR```
* Add installation path to your ```PYTHONPATH``` environment variable:   
  ``` export PYTHONPATH=$PYTHONPATH:$INSTALL_DIR/lib/python3.6/site-packages ```
* Build and install package:   
  ``` python3.6 setup.py sdist bdist_wheel```    
  ``` python3.6 setup build```   
  ``` python3.6 setup install --prefix=$INSTALL_DIR```   

All dependencies will be automatically downloaded and installed in ```$INSTALL_DIR```.   
     
To use package scripts:

* Add binary directory to your ```PATH``` environment variable:   
  ``` export PATH=$PATH:$INSTALL_DIR/bin ```    

## **How to run the service?**  

In the following we describe the steps done to deploy and run the application and the auxiliary services. Three possible options are described below for the deployment, depending of whether the job management is done with celery, Kubernetes, or with Slurm. To ease the deployment we provide Docker containers and configuration files for Docker Compose or Kubernetes.       

### **Preliminary setup**
Before running the application you must do some preparatory stuff:   

* (OPTIONAL) Create a dedicated user & group (e.g. `caesar`) allowed to run the application and services and give it ownership to the directories created below    * Create the application working dir (by default `/opt/caesar-rest`)   
* (OPTIONAL) Mount an external storage in the application working dir, for example using rclone: `/usr/bin/rclone mount --daemon [--uid=[UID] --gid=[UID]] --umask 000 --allow-other --file-perms 0777 --dir-cache-time 0m5s --vfs-cache-mode full [RCLONE_REMOTE_STORAGE]:[RCLONE_REMOTE_STORAGE_PATH] /opt/caesar-rest -vvv` where `UID` is the Linux user id of the user previously created.     
* Create the top directory for data upload (by default `/opt/caesar-rest/data`)   
* Create the top directory for jobs (by default `/opt/caesar-rest/jobs`)   
* (OPTIONAL) Create the log directory for system services (see below), e.g. `/opt/caesar-rest/logs` 
* (OPTIONAL) Create the run directory for system services (see below), e.g. `/opt/caesar-rest/run` 


### **Run DB service**
caesar-rest requires a MongoDB service where to store user data and job information. To start the DB service:    

```systemctl start mongodb.service```   

Alternatively you can use the Docker container ```sriggi/caesar-rest-db:latest``` (see https://hub.docker.com/r/sriggi/caesar-rest-db) and deploy it with DockerCompose or Kubernetes (see the configuration files under the repository ```config``` directory.   

### **Run Filebeat service**   
caesar-rest uses filebeat to forward file logs to an ElasticSearch service. To start the service:   

```systemctl start filebeat.service```    

Alternatively, you can use the Docker container for the application ```sriggi/caesar-rest:latest``` (see https://hub.docker.com/r/sriggi/caesar-rest) setting the container option ```FORWARD_LOGS=1```. This will start the filebeat service in the web application container.   

### **Run Celery services (OPTIONAL)**
If you want to manage jobs with Celery, you must run a message broker service (i.e. rabbitmq), a task store service (i.e. redis or mongdb) and one or more Celery worker services.   

#### **Run broker service**   
To run the rabbimq message broker service:   
   
```systemctl start rabbitmq-server.service```   

Alternatively, you can use the Docker container ```sriggi/caesar-rest-broker:latest``` (see https://hub.docker.com/r/sriggi/caesar-rest-broker) and deploy it with DockerCompose or Kubernetes (see the configuration files under the repository ```config``` directory.      
   
#### **Run task store service**   
If you have chosen MongoDB as task store, you are already running the service (see previous section `Run DB service`). However, if you want to use Redis as task store, run it as follows:       
  
```systemctl start redis.service```    

Docker container is still to be produced.   
   
#### **Run celery workers**   
Run celery worker with desired concurrency level (e.g. 2), message queue (e.g. celery), broker and result backend urls:  
   
```celery --broker=[BROKER_URL] --result-backend=[RESULT_BACKEND_URL] --app=caesar_rest worker --loglevel=INFO --concurrency=2 -Q celery```   
   
In production you may want to run this as a system service:   
       
* Create a `/etc/default/caesar-workers` configuration file (e.g. see the example in the `config/celery` directory):  
   
     ```
     # The names of the workers. Only one here. 
     CELERYD_NODES="caesar_worker"    
     
     # The name of the Celery App   
     CELERY_APP="caesar_rest"
      
     # Working dir    
     CELERYD_CHDIR="/opt/caesar-rest"    
     
     # Additional options    
     CELERYD_OPTS="--time-limit=300 --concurrency=4"

     # Log and PID directories    
     CELERYD_LOG_FILE="/opt/caesar-rest/logs/%n%I.log"    
     CELERYD_PID_FILE="/opt/caesar-rest/run/%n.pid"    

     # Log level    
     CELERYD_LOG_LEVEL=INFO    

     # Path to celery binary, that is in your virtual environment    
     CELERY_BIN=/usr/local/bin/celery    
     ```
     
* Create a `/etc/systemd/system/caesar-workers.service` systemd service file:    
   
     ```
     [Unit]    
     Description=Caesar Celery Worker Service    
     After=network.target rabbitmq-server.target redis.target   

     [Service]    
     Type=forking   
     User=caesar   
     Group=caesar   
     EnvironmentFile=/etc/default/caesar-workers     
     Environment="PATH=$INSTALL_DIR/bin"   
     Environment="PYTHONPATH=$INSTALL_DIR/lib/python2.7/site-packages"   
     WorkingDirectory=/opt/caesar-rest   
     ExecStart=/bin/sh -c '${CELERY_BIN} multi start ${CELERYD_NODES} \    
       -A ${CELERY_APP} --pidfile=${CELERYD_PID_FILE} \   
       --logfile=${CELERYD_LOG_FILE} --loglevel=${CELERYD_LOG_LEVEL} ${CELERYD_OPTS}'    
     ExecStop=/bin/sh -c '${CELERY_BIN} multi stopwait ${CELERYD_NODES} \    
       --pidfile=${CELERYD_PID_FILE}'   
     ExecReload=/bin/sh -c '${CELERY_BIN} multi restart ${CELERYD_NODES} \   
       -A ${CELERY_APP} --pidfile=${CELERYD_PID_FILE} \   
       --logfile=${CELERYD_LOG_FILE} --loglevel=${CELERYD_LOG_LEVEL} ${CELERYD_OPTS}'    

     [Install]    
     WantedBy=multi-user.target   
     ```
  
* Start the service:   
  
     ```sudo systemctl caesar-workers.service start```    
     
Alternatively, you can use the Docker container ```sriggi/caesar-rest-worker:latest``` (https://hub.docker.com/r/sriggi/caesar-rest-worker) and deploy it with DockerCompose or Kubernetes (see the configuration files under the repository ```config``` directory.      
   
### **Run the web application**   

#### **Run the application in development mode**   
To run caesar-rest in development mode, e.g. for debug or testing purposes:   

  ```$INSTALL_DIR/bin/run_app.py --[ARGS]```

where supported `ARGS` are:    

   MAIN OPTIONS       
   * `datadir=[DATADIR]`: Directory where to store uploaded data (default: /opt/caesar-rest/data)   
   * `jobdir=[JOBDIR]`: Top directory where to store job data (default: /opt/caesar-rest/jobs)     
   * `job_scheduler=[SCHEDULER]`:  Job scheduler to be used. Options are: {celery,kubernetes,slurm} (default=celery)     
   * `debug`: Run Flask application in debug mode if given   
   * `ssl`: To enable run of Flask application over HTTPS     

   AAI OPTIONS
   * `aai`: Enable service authentication    
   * `secretfile=[SECRETFILE]`: File (.json) with OpenID Connect client auth credentials    
   
   DB OPTIONS       
   * `dbname=[DBNAME]`: Name of MongoDB database (default=caesardb)   
   * `dbhost=[DBHOST]`: Host of MongoDB database (default=localhost)    
   * `dbport=[DBPORT]`: Port of MongoDB database (default=27017)   

   LOGGING OPTIONS
   * `loglevel=[LEVEL]`: Log level to be used (default=INFO)   
   * `logtofile`: Enable logging to file (default=no)   
   * `logdir`: Directory where to store logs (default=/opt/caesar-rest/logs)   
   * `logfile`: Name of json log file (default=app_logs.json)   
   * `logfile_maxsize`: Max file size in MB (default=5)    
  
   CELERY OPTIONS       
   * `result_backend_host=[BACKEND_HOST]`: Host of Celery result backend service (default=localhost) 
   * `result_backend_port=[BACKEND_PORT]`: Port of Celery result backend service (default=6379)   
   * `result_backend_proto=[BACKEND_PROTO]`: Celery result backend type. Options are: {mongodb,redis} (default=redis)   
   * `result_backend_dbname=[BACKEND_DBNAME]`: Celery result backend database name (default=0)   
   * `broker_host=[BROKER_HOST]`: Host of Celery broker service (default=localhost)    
   * `broker_port=[BROKER_PORT]`: Port of Celery broker service (default=5672)    
   * `broker_proto=[BROKER_PROTO]`: Protocol of Celery broker. Options are: {amqp,redis} (default=amqp)    
   * `broker_user=[BROKER_USER]`: Username used in Celery broker (default=guest)   
   * `broker_pass=[BROKER_PASS]`: Password used in Celery broker (default=guest)   
  
   KUBERNETES OPTIONS   
   * `kube_config=[FILE_PATH]`: Kube configuration file path (default=search in standard path)   
   * `kube_cafile=[FILE_PATH]`: Kube certificate authority file path    
   * `kube_keyfile=[FILE_PATH]`: Kube private key file path    
   * `kube_certfile=[FILE_PATH]`: Kube certificate file path   

   SLURM OPTIONS   
   * `slurm_keyfile=[FILE_PATH]`: Slurm rest service private key file path    
   * `slurm_user=[SLURM_USER]`: Username enabled to run in Slurm cluster (default=cirasa)   
   * `slurm_host=[SLURM_HOST]`: Slurm cluster host/ipaddress (default=localhost)   
   * `slurm_port=[SLURM_PORT]`: Slurm rest service port (default=6820)  
   * `slurm_batch_workdir=[SLURM_BATCH_WORKDIR]`: Cluster directory where to place Slurm batch logs (must be writable by slurm_user) (default=/opt/slurm/batchlogs/caesar-rest)    
   * `slurm_queue=[SLURM_QUEUE]`: Slurm cluster queue for submitting jobs (default=normal)   
   * `slurm_jobdir=[SLURM_JOBDIR]`: Path at which the job directory is mounted in Slurm cluster (default=/mnt/storage/jobs)    
   * `slurm_datadir=[SLURM_DATADIR]`: Path at which the data directory is mounted in Slurm cluster (default=/mnt/storage/data)   
   * `slurm_max_cores_per_job=[SLURM_MAX_CORES_PER_JOB]`: Slurm maximum number of cores reserved for a job (default=4)   
    
   VOLUME MOUNT OPTIONS   
   * `mount_rclone_volume`: Enable mounting of Nextcloud volume through rclone in container jobs (default=no)  
   * `mount_volume_path=[PATH]`: Mount volume path for container jobs (default=/mnt/storage)  
   * `rclone_storage_name=[NAME]`: rclone remote storage name (default=neanias-nextcloud)   
   * `rclone_storage_path=[PATH]`: rclone remote storage path (default=.)    	
  
Flask default options are defined in the `config.py`. Celery options are defined in the `celery_config.py`. Other options may be defined in the future to override default Flask and Celery options.   

#### **Run the application in production**   
In a production environment you can run the application behind a nginx+uwsgi (or nginx+gunicorn) server. In the `config` directory of the repository you can find sample files to create and configure required services. For example:  

* Start the application with uwsgi:   
     
  uwsgi --wsgi-file $INSTALL_DIR/bin/run_app.py --callable app [WSGI_CONFIG_FILE]

  where ```WSGI_CONFIG_FILE``` is a configuration file (.ini format) for uwsgi. A sample configuration file is provided in the `config/uwgsi` directory:   
  
  ```
  [uwsgi]
  processes = 4   
  threads = 2   
  socket = ./run/caesar-rest.sock   
  ;socket = :5000
  ;http-socket = :5000
  socket-timeout = 65
  
  buffer-size = 32768  
  master = true   
  chmod-socket = 660   
  vacuum = true  
  die-on-term = true  
  ```
  
  Alternatively you can configure options from command line, e.g.:    
  
   ```uwsgi --uid=[RUNUSER] --gid=[RUNUSER] --binary-path /usr/local/bin/uwsgi --wsgi-file=$INSTALL_DIR/bin/run_app.py --callable=app --pyargv=[APP_ARGS] --workers=[NWORKERS] --enable-threads --threads=[NTHREADS] --http-socket="0.0.0.0:[PORT]" --http-timeout=[SOCKET_TIMEOUT] --http-enable-proxy-protocol --http-auto-chunked --socket-timeout=[SOCKET_TIMEOUT] --master --chmod-socket=660 --chown-socket=[RUNUSER] --buffer-size=[BUFFER_SIZE] --vacuum --die-on-term ```
  
  where ```APP_ARGS``` are the application command line options described in the previous paragraph and ```RUNUSER``` is the username chosen for running the service. The other options are described in the uwsgi online documentation.    
  
  In production you may want to run this as a system service: 
  
  - Create an `/etc/systemd/system/caesar-rest.service` systemd service file, for example following the example provided in the `config/uwsgi` directory:       
       
    ```
    [Unit]
    Description=uWSGI instance to serve caesar-rest application    
    After=network.target caesar-workers.target   

    [Service]
    User=caesar  
    Group=www-data   
    WorkingDirectory=/opt/caesar-rest  
    Environment="PATH=$INSTALL_DIR/bin"   
    Environment="PYTHONPATH=$INSTALL_DIR/lib/python2.7/site-packages"  
    ExecStart=/usr/bin/uwsgi --wsgi-file $INSTALL_DIR/bin/run_app.py --callable app --ini /opt/caesar-rest/config/uwsgi.ini

    [Install]   
    WantedBy=multi-user.target    
    ```   
    
   - Start the service:   
     ```sudo systemctl caesar-rest.service start```    

   Alternatively, you can use the Docker container `sriggi/caesar-rest:devel` (see https://hub.docker.com/r/sriggi/caesar-rest) and deploy it with DockerCompose or Kubernetes (see the configuration files under the repository ```config``` directory. All application command line options described in the previous section can be configured from container env variables.        

* Start the nginx service:

  - Create a `/etc/nginx/conf.d/nginx.conf` configuration file (see example file provided in the `config/nginx` directory):      

    ```
    server {   
      listen 8080;   
      client_max_body_size 1000M;   
      sendfile on;    
      keepalive_timeout 0;   
      location / {   
        include uwsgi_params;    
        uwsgi_pass unix:/opt/caesar-rest/run/caesar-rest.sock;   
      }       
    }    
    ```
  
    With this sample configuration the nginx server will listen at port 8080 and call the caesar-rest application via socket. An alternative configuration could be:    
    
    ```
    upstream backend {
      least_conn;  # load balancing strategy
      server [HOST1]:[PORT];
      server [HOST1]:[PORT];
      keepalive 64;
    }

    server {
      listen 8080;
      client_max_body_size 1000M;
      large_client_header_buffers 4 32k;
      sendfile on;
      keepalive_timeout 0;
      location / {
        include uwsgi_params;
        uwsgi_pass backend;
      }
    }
    ```
    
    with nginx load balancing incoming requests, sending them to 2 caesar-rest http applications listening at `HOST1` and `HOST2` on port `PORT`.    
   
  - Create a `/etc/systemd/system/nginx.service` systemd file, e.g. see the example provided in the `config/nginx` directory:   
  
    ```
    [Unit]   
    Description=The NGINX HTTP and reverse proxy server  
    After=syslog.target network.target remote-fs.target nss-lookup.target caesar-rest.target   

    [Service]   
    Type=forking    
    PIDFile=/run/nginx.pid   
    ExecStartPre=/usr/sbin/nginx -t   
    ExecStart=/usr/sbin/nginx   
    ExecReload=/usr/sbin/nginx -s reload   
    ExecStop=/bin/kill -s QUIT $MAINPID   
    PrivateTmp=true    

    [Install]   
    WantedBy=multi-user.target   
    ```
  
  - Run nginx server:   

    ```sudo systemctl start nginx.service```

  Alternatively you can use the Docker container `sriggi/caesar-rest-lb:latest` (see https://hub.docker.com/r/sriggi/caesar-rest-lb) and deploy it with DockerCompose. In Kubernetes this functionality is provided by ingresses (see sample configuration files).   

### **Run job monitoring service**   
The job monitoring service periodically monitors user jobs, updating their status on the DB. It can be started as:    

```$INSTALL_DIR/bin/run_jobmonitor.py --[ARGS]```    

where supported `ARGS` are:   

   * `job_monitoring_period=[PERIOD]`: Job monitoring poll period in seconds (default=30)     
   * `job_scheduler=[SCHEDULER]`:  Job scheduler to be used. Options are: {celery,kubernetes,slurm} (default=celery)     
   * `dbname=[DBNAME]`: Name of MongoDB database (default=caesardb)   
   * `dbhost=[DBHOST]`: Host of MongoDB database (default=localhost)    
   * `dbport=[DBPORT]`: Port of MongoDB database (default=27017)      
   * `kube_config=[FILE_PATH]`: Kube configuration file path (default=search in standard path)   
   * `kube_cafile=[FILE_PATH]`: Kube certificate authority file path    
   * `kube_keyfile=[FILE_PATH]`: Kube private key file path    
   * `kube_certfile=[FILE_PATH]`: Kube certificate file path   
   * `slurm_keyfile=[FILE_PATH]`: Slurm rest service private key file path    
   * `slurm_user=[SLURM_USER]`: Username enabled to run in Slurm cluster (default=cirasa)   
   * `slurm_host=[SLURM_HOST]`: Slurm cluster host/ipaddress (default=localhost)   
   * `slurm_port=[SLURM_PORT]`: Slurm rest service port (default=6820)  

Alternatively, you can use the Docker container `sriggi/caesar-rest-jobmonitor:latest` (see https://hub.docker.com/r/sriggi/caesar-rest-jobmonitor) and deploy it with DockerCompose or Kubernetes (see sample configuration files).    
   
### **Run accounting service**   
The accounting service periodically monitors user data and job info, storing aggregated stats in the DB. It can be started as:    

```$INSTALL_DIR/bin/run_accounter.py --[ARGS]```    

where supported `ARGS` are:   

   * `datadir=[DATADIR]`: Directory where to store uploaded data (default: /opt/caesar-rest/data)   
   * `jobdir=[JOBDIR]`: Top directory where to store job data (default: /opt/caesar-rest/jobs)     
   * `job_monitoring_period=[PERIOD]`: Job info monitoring poll period in seconds (default=30) 
   * `dbname=[DBNAME]`: Name of MongoDB database (default=caesardb)   
   * `dbhost=[DBHOST]`: Host of MongoDB database (default=localhost)    
   * `dbport=[DBPORT]`: Port of MongoDB database (default=27017)      
   * `mount_rclone_volume`: Enable mounting of Nextcloud volume through rclone in container jobs (default=no)  
   * `mount_volume_path=[PATH]`: Mount volume path for container jobs (default=/mnt/storage)  
   * `rclone_storage_name=[NAME]`: rclone remote storage name (default=neanias-nextcloud)   
   * `rclone_storage_path=[PATH]`: rclone remote storage path (default=.)   

Alternatively, you can use the Docker container `sriggi/caesar-rest-accounter:latest` (see https://hub.docker.com/r/sriggi/caesar-rest-accounter) and deploy it with DockerCompose or Kubernetes (see sample configuration files).    

## **Usage**  
caesar-rest provides the following REST endpoints:   

### **Data upload**

* URL:```http://server-address:port/caesar/api/v1.0/upload```   
* Request methods: POST   
* Request header: ```content-type: multipart/form-data```   

A sample curl request would be:   

```
curl -X POST \   
  -H 'Content-Type: multipart/form-data' \   
  -F 'file=@VGPS_cont_MOS017.fits' \   
  --url 'http://localhost:8080/caesar/api/v1.0/upload'   
```

Server response is:   
```
{
  "date":"2020-04-24T17:04:26.174333",
  "filename_orig":"VGPS_cont_MOS017.fits",
  "format":"fits",
  "size":4.00726318359375,
  "status":"File uploaded with success",
  "uuid":"250fdf5ed6a044888cf4406338f9e73b"
}
```

A file uuid (or file path) are returned and can be used to download the file or set job input file information.   

### **Data download**

* URL:```http://server-address:port/caesar/api/v1.0/download/[file_id]```   
* Request methods: GET, POST   
* Request header: None  

A sample curl request would be:   

```
curl  -X GET \
  --fail -o data.fits \
  --url 'http://localhost:8080/caesar/api/v1.0/download/67a49bf7555b41739095681bf52a1f99'
```

The above request will fail if file is not found, otherwise the downloaded file will be saves as `data.fits`. Without the `-o` argument raw output is written to stdout. If file is not found a json response is returned:   

```
{
  "status": "File with uuid 67a49bf7555b41739095681bf52a1f99 not found on the system!"
}
```

### **Get uploaded data ids**

* URL:```http://server-address:port/caesar/api/v1.0/fileids```   
* Request methods: GET   
* Request header: None  

A sample curl request would be:   

```
curl  -X GET \
  --url 'http://localhost:8080/caesar/api/v1.0/fileids'
```

with response:    

```
{"file_ids":["a668c353ba4d4c7395ad94b4e8647d92","c54db5ef95734c62a499db38587c48a5","26bc9a545c8f4f05a2c719ec5c3917e0"]}
```

### **App description**
To get the list of supported apps:   

* URL:```http://server-address:port/caesar/api/v1.0/apps```   
* Request methods: GET   
* Request header: none

Server response contains a list of valid apps that can be queried for further description and used in job submission:    

```
{
  "apps": [
    "caesar",
    "mrcnn"
  ]
}
```

To get information about a given app:  

* URL:```http://server-address:port/caesar/api/v1.0/app/[app_name]/describe```   
* Request methods: GET    
* Request header: none

Server response contains a list of app options that can be used in job submission:   

```
{"image":{"description":"Path to input image (.fits) to be given to classifier (default=empty)","mandatory":true,"type":"str"},"iouThr":{"description":"IOU threshold between detected and ground truth bboxes to consider the object as detected (default=0.6)","mandatory":false,"type":"float"},"scoreThr":{"description":"Detected object score threshold to select as final object (default=0.7)","mandatory":false,"type":"float"}}
```

### **Job submission**
* URL:```http://server-address:port/caesar/api/v1.0/job```   
* Request methods: POST   
* Request header: ```content-type: application/json```   

A sample curl request would be:   

```
curl -X POST \   
  -H 'Content-Type: application/json' \   
  -d '{"app":"caesar","job_inputs":{"inputfile":"/opt/caesar-rest/data/67a49bf7555b41739095681bf52a1f99.fits","run":true,"no-logredir":true,"envfile":"/home/riggi/Software/setvars.sh","no-mpi":true,"no-nestedsearch":true,"no-extendedsearch":true}}' \   
  --url 'http://localhost:8080/caesar/api/v1.0/job'   
```

Job data must contain a valid app name (in this case `caesar`) and desired job inputs, e.g. a dictionary with app valid options. Valid options for `caesar` app are named as in `caesar` and can be retrieved using app description url described above.   

Server response is:   

```
{
  "app": "caesar",
  "job_id": "69ca62d7-5098-4fe7-a675-63895a2d06b1",
  "job_inputs": {
    "envfile": "/home/riggi/Software/setvars.sh",
    "inputfile": "67a49bf7555b41739095681bf52a1f99",
    "no-extendedsearch": true,
    "no-logredir": true,
    "no-mpi": true,
    "no-nestedsearch": true,
    "run": true
  },
  "status": "Job submitted with success",
  "submit_date": "2020-04-24T14:05:24.761766"
}
```

A job id is returned in the response which can be used to query the status of the job or cancel it or retrieve output data at completion. 

### **Get job status**
* URL:```http://server-address:port/caesar/api/v1.0/job/[job_id]/status```   
* Request methods: GET   
* Request header: None   

A sample curl request would be:   

```
curl -X GET \   
  --url 'http://localhost:8080/caesar/api/v1.0/job/f135bcee-562b-4f01-ad9b-103c35b13b36/status'   
```

Server response is:   

```
{
  "elapsed_time": "27.3435878754",
  "exit_status": 0,
  "job_id": "f135bcee-562b-4f01-ad9b-103c35b13b36",
  "pid": "11539",
  "state": "SUCCESS",
  "status": "Process terminated with success"
}
```

Exit status is the shell exit status of background task executed and pid the corresponding process id. Possible job states are: {STARTED, TIMED-OUT, ABORTED, RUNNING, SUCCESS, FAILURE}. 


### **Get job output**
* URL:```http://server-address:port/caesar/api/v1.0/job/[job_id]/output```   
* Request methods: GET   
* Request header: None   

A sample curl request would be:   

```
curl -X GET \   
  --fail -o job_output.tar.gz \
  --url 'http://localhost:8080/caesar/api/v1.0/job/c3c9348a-bea0-4141-8fe9-7f64076a2327/output'   
```

The response is a tar.gz file containing all job directory files (logs, output data, run scripts, etc).  

### **Cancel job**
* URL:```http://server-address:port/caesar/api/v1.0/job/[job_id]/cancel```   
* Request methods: POST   
* Request header: None  

### **Get job ids**
* URL:```http://server-address:port/caesar/api/v1.0/jobs```   
* Request methods: GET   
* Request header: None  
