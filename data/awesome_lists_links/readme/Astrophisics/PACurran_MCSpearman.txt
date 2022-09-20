MCSpearman
==========

A program to compute the Spearman's rank correlation coefficient, including a Monte Carlo error analysis. Method is detailed at http://arxiv.org/abs/1411.3816 and Astrophysics Source Code Library record is at http://ascl.net/1504.008. 

Prerequisites:

1. C compiler (e.g., gcc)
2. GNU Scientific Library (GSL) development & library packages


Notes on GNU Scientific Library (GSL): GSL can be downloaded from
www.gnu.org/software/gsl/, or installed directly from the Ubuntu
Software Centre.


Installation (assuming terminal): 

1. Modify compile.sh so it points to local GSL directory. Can find this via 
> gsl-config --libs

2. Modify permissions of compile.sh
> chmod a+x compile.sh 

3. Run compile.sh 
> ./compile.sh 

4. Run MCSpearman for help
> ./mcspearman -h

