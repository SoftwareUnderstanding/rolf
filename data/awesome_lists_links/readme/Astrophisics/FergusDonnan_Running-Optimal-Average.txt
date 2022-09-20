# Running-Optimal-Average

The running optimal average is a smooth/flexible model that can describe time series data. This uses a Gaussian window function that moves through the data giving stronger weights to points close to the centre of the Gaussian. Therefore the width of the window function, delta, controls the flexibility of the model, with a small delta providing a very flexible model.

The function also calculates the effective no. of parameters as a very flexible model will correspond to large no. of parameters while 
a rigid model (low delta) has a low effective no. of parameters.

An error envelope is also calculated for the model.

## Installation

install using pip:
pip install ROA





## Usage:

t, model, errs, P = RunningOptimalAverage(t_data, Flux, Flux_err, delta)

Calculate running optimal average on a fine grid of 1000 equally spaced points over the range of data. Also returns errors and effective number of parameters.

Import using:

from ROA import RunningOptimalAverage

Parameters
----------
t_data  :  float array :
    Time values of the data points
    
Flux  : float array :
    Flux data values
    
Flux_err : float array :
    Errors for the flux data points
    
delta  : float :
    Window width of the Gaussian memory function - controls how flexible the model is


Returns
----------
t  : float array :
    Time values of the grid used to calculate ROA
    
model : float array :
    Running optimal average's calculated for each time t
    
errs : float array :
    Errors of the running optimal average
    
P : float :
    Effective number of parameters for the model




