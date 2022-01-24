# Machine Learning 

This library contains functionality for dealing with datasets used for Machine Learning.

`Install-Package ZeroProximity.MachineLearning`

## Imbalanced Data

Implements SMOTE for imbalanced datasets
Based on pseudocode from the whitepaper: https://arxiv.org/pdf/1106.1813.pdf

```csharp
var balanced = Smote.Balance(data, labels);
```

## Nearest Neighbours

Implements a simple Nearest Neighbour function that returns the indexes in a dataset that 
represent the closest matches.

## Matrix Normalization

Implements a simple function to normalize data in a matrix,
this is required for some Linear Regression, Neural Net and Nearest Neighbour algorithms.
