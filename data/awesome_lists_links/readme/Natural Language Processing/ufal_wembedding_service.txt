# WEmbedding service

This service encodes given text to a word embedding vector from supported
language representation model. We currently support multilingual BERT
(https://arxiv.org/abs/1810.04805) computed by Transformers
(https://arxiv.org/abs/1910.03771, https://github.com/huggingface/transformers).

The service can be used as a module or run as a server to handle queries.

## License

Copyright 2021 Institute of Formal and Applied Linguistics, Faculty of Mathematics and Physics, Charles University, Czech Republic.

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

## Author and Contact

Author: Milan Straka
Contact: straka@ufal.mff.cuni.cz

## Installation

Clone the repository:

```sh
git clone https://github.com/ufal/wembedding_service
```

Create a Python virtual environment:

```sh
python -m venv venv
```

Install requirements:

```sh
venv/bin/pip3 install -r requirements.txt
```

Run the service:

```sh
venv/bin/python3 ./start_wembeddings_server.py 8000
```

## Docker

Build the Docker image:

```sh
docker build -t wembeddings
```

Run the container:

```sh
docker run --name wembeddings --rm wembeddings 8000
```

If you wish to run the service in the background, you can add the `-d` option:

```sh
docker run --name wembeddings --rm wembeddings -d --rm wembeddings 8000
```

If you have another Docker container to query the WEmbeddings service, you can
connect both the service and the client to a user-defined network:

```sh
docker network create wembeddings-network
docker run --network wembeddings-network --name wembeddings -d --rm wembeddings 8000
docker run --network wembeddings-network --name my_client --rm my_client
```

The client `my_client` can access the wembeddings service by its assigned Docker
name `wembeddings:8000`.
