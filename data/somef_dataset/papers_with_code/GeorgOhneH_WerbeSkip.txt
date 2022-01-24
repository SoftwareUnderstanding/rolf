# WerbeSkip
A Webserver to detect ads on TV

Build with Neural Networks, Django and Vue

## Installation
Note: Cloning this repository may take a while, because it's very large

It's only testet with python 3.6, other versions may not work properly
All dependencys for python are in `requirements.txt` listed.

If you want to use the webserver, [Redis](https://redis.io/download)
must also be installed.

### External Files
So that evrything works correctly the dateset and the saved network,
must be downloaded. For the dateset go [here](helperfunctions/prosieben)
and for the network go [here](helperfunctions/prosieben/networks/teleboy)

# Usage
## Neuron Network
Go to the directory [deepnet/examples](deepnet/examples) for examples.
## Webserver
To start the server run `sh deploy.sh`.

If you prefer docker run `docker-compose up`

# File Structure
There are 2 parts to the app. The first part is the neural network and
the algorithmen. The second part is the webserver.
## Neural Network
The main part is the deepnet directory. It holds a self written deep
neural network library.

The numpywrapper module is just so the neural network library can
switch between numpy and cupy

## Webserver
The main parts are: app, src, vuedj

vuedj: The Django settings

app: Backend with Websocket

src: Frontend implemented with VueJS

## Others
thoughts is the directory with the Protocoll and the maturaarbeit

update_handler.py is the intersection between the server and the algorithmen
