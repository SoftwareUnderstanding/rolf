# BANG 
## BAyesian decomposiotioN of Galaxies

BANG is a GPU/CPU-python code for modelling both the photometry and kinematics of galaxies.
The underlying model is the superposition of different component the user has 3 possible 
combination:

* Bulge + inner disc + outer disc + Halo
* Bulge +  disc  + Halo
* inner disc + outer disc + Halo

For any detail about the model construction see Rigamonti et al. 2022.

The parameter estimation is done with a python implementation [CPnest](https://github.com/johnveitch/cpnest) 
of nested sampling algorithm.

We strongly suggest to run BANG on GPU. CPU parameter estimation can take
days. A fast CPU implementation will be available in a future release of the code.


All the function needed by the user are well documented. In order to run BANG on 
your galaxy open the example.py script from the BANG/src/BANG or BANG/test directories
and follow the instructions.

Once your data have been correctly prepared and the config.yaml file has been created, 
running BANG requires few lines of code.


For any problem or suggestion feel free to contact the authors at:

            frigamonti@uninsubria.it

For installing BANG you can follow these instructions:

1- Download the BANG package from github. You can simply type from your terminal: 
            git clone https://github.com/FabioRigamonti/BANG.git

2- Instal the python modules with: 
            pip install BANGal

3- Copy the files:
        setup_easy.pyx
        utils_easy.pyx
        dehnen/
    From the github repo (you can find them in BANG/src/BANG) to the directory
    where pip has installed BANG. You can find this directory by opening a python shell
    importing BANG and printing BANG.

4- Move to the directory where pip has installed BANG and run the following command:
        python setup_easy.py build_ext --inplace


    



