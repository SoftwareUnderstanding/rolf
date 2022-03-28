

 # TicTacToe
This trains agent  to play TicTac via reinforcement learnig. 

To Train:-  Run main.py.  It will output Q<index> file which contains Q value function, and other plots and file. One set of output file indexed with 129 is provided.

To change parameters for the training :- edit parameters values in pm.py

To play with trained agent interactively:- 1) Run play.py, 2)enter 'Q<>' with which you want to play. ex Q10, Q129 (see
     which file was created.   3) Enter yes if want to play first , no if you want agent to play first.
     You must have opencv installed for graphics to run.



  # ALPHA GO 

nn.py has neural network library, like linear, relu, softmax, convolution, cre, mse forward and backpropagation.
For theoritical proofs see nn.pdf.
          # Backprop for convolution (1d input & 1d output,  2d input &2d weight)
input x,output y, filter w, inward grad dy, outward grad dx, mode {'full', 'valid', same','custom'}. Algorithm is below.

X = pad(x, required for mode)

y = correlate(X,w, 'valid')

dX = convolve(dy, w, 'full')

dw = correlate(X,dy, 'valid')

dx = uppad(dX, same as padding)

         #Backprop for convolution( 3d input 4d weights)
dim(x) = (l,m,n), dim(w) = (k,l,m,n) , dim(dy) = (k,m*,n*)

X = pad(x, as required for mode), # dim(X) = (l,m',n')  

Y = [[correlate(x2,w2,mode='valid') for x2,w2 in zip(X, w1)] for w1 in w]) # dim(Y) = (k,l,m*,n*)  

y = sum(Y, axis=1) # dim(y) = (k,m*,n*)

dX = [[convolve(dy,w1,mode='full') for w1 in w] for dy,w in zip(dy,w)]) #padded dX, dim(k,l,m',n')

dw = [[correlate(X, dy, mode='valid') for X in X] for dy in dy])     # dim(k,l,m,n)

dX1 = sum(self.dX, axis=0)  #padded dx, dim(l,m',n')

dx =  unpad(dX1)  #dim(l,m,n)


               # Opitimizer ADAM
See original paper ADAM: A METHOD  FOR STOCHASTIC OPTIMIZATION for excellent reference.
https://arxiv.org/pdf/1412.6980.pdf      

Various learning rate setting is employed in optimization of deep learning architechture. See sa.pdf for some experiments and results.
