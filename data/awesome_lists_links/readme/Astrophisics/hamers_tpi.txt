# Test Particle Integrator (TPI)

Adrian Hamers, based on Hamers et al. (2014; http://adsabs.harvard.edu/abs/2014MNRAS.443..355H)
    
A code to compute the gravitational dynamics of particles orbiting a supermassive black hole (SBH). A distinction is made to two types of particles: test particles and field particles. Field particles are assumed to move in quasi-static Keplerian orbits around the SBH that precess due to the enclosed mass (Newtonian `mass precession') and relativistic effects. Otherwise, field-particle-field-particle interactions are neglected. Test particles are integrated in the time-dependent potential of the field particles and the SBH. Relativistic effects are included in the equations of motion (including the effects of SBH spin), and test-particle-test-particle interactions are neglected. Supports OpenMP; legacy GPU support (not tested recently).
    
Compilation: use the makefile

Simple usage: ./tpi -i input/run1.txt 

