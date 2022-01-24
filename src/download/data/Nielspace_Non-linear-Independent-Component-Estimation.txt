NICE:Non-Linear Independent Component Estimation

A new deep learning framework technique was proposed for modeling complex high-dimensional densities called Non-linear Independent Component Estimation (NICE). The idea behind this framework is that we create a deep learning model which can read a high dimensional data and which is easy to represent. Using this model we can easily map data which is high in complexity and dimensionality. For this purpose, a non-linear deterministic transformation of the data is learned that maps it to a latent space so as to make the transformed data conform to a factorized distribution resulting in independent latent variable. 

Various methods has been used to implement this technique. 

1. TRIANGULAR STRUCTURE -- to obtain a family of bijections whose Jacobian determinant is tractable and whose computation is straightforward, both forwards (the encoder f ) and backwards (the decoder f−1). 

2. COUPLING LAYER -- a family of bijective transformation with triangular Jacobian therefore
tractable Jacobian determinant. That will serve a building block for the transformation f .

3. RESCALING -- Using PCA: Principle Component Analysis as well ZCA: Zero Component Analysis

4. LOG-LIKELIHOOD AND GENERATION

In this project I have tried to achieve the same results in my own way. 

PS. Please read the proposed paper from the given link "https://arxiv.org/pdf/1410.8516.pdf"