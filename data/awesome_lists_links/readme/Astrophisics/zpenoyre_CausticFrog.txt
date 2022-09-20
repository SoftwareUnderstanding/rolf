# CausticFrog
Simulation specifically designed to model the reaction of a system of orbiting particles to instantaneous mass loss. For more details refer to Penoyre & Haiman 2017 (https://ui.adsabs.harvard.edu/#abs/2018MNRAS.473..498P/abstract)

The code applies to any spherically symmetric potential, and follows the radial evolution of shells of mass. The code tracks the inner and outer edge of each shell, whose radius evolves as a test particle. The amount of mass in each shell is fixed but multiple shells can overlap leading to higher densities. 

The code performs leapfrog integration on these edges, recaluclating the enclosed mass (including self-gravity of all enclosed shells) of each edge at every time-step.

The reason the code is set-up as such is to capture behaviour that is easily lost in n-body simulations or similar: that high densities appear not just where many shells overlap, but also where edges of individual shells overlap and the volume of the shell goes to 0.

The shells are attached, edge to edge, in a "concertina" which spans all radii and cuts computation cost. Many concertinas can be evolved at once, and for systems with many initial eccentricities and phases each concertina correspondons to a single initial state.

User-specified mass profiles can be used to set-up any spherical potential and initial state.

It is written in Python and Cython, with most of the heavy lifting falling on a Cython function finding the mass internal to an edge. It can be run on a personal computer with one timestep taking ~1 second for around 10,000 shells. For many more shells the performance drops.

An example iPython notebook is included showing some of the functionality. In all honesty, if you'd like to use it for any complex task I recommend contacting me (at zpenoyre@astro.columbia.edu). It is relatively short and thus should be easy to further adapt, optimize and build from.
