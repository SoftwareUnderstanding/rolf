# rgpy
## The Renormalization Group and Machine Learning
As part of my capstone (bachelor's thesis) at Amsterdam University College, I am conducting research regarding the link between machine learning and statistical physics, more specifically, the renormalization group.

In this repository, I aim to offer a set of tools to researchers hoping to use ML in physics investigations, specifically to calculate critical exponents for interesting physical systems. As a bonus, I'll add some standard renormalization techniques.

The project is structured as follows:

> samplers
  - Generation of samples through MCMC techniques (in both tensorflow and numpy implementations):
    - Metropolis-Hastings (tf, np)
    - Swendsen-Wang (np)
    - Wolff (np)

> rbms
  - Restricted Boltzmann Machines (RBM)::
    - Contrastive-divergence (both bernoulli- and binary-valued)
    - Real-space mutual information maximization (from Koch-Janusz and Ringel)

> standard
  - Majority-rule block-spin renormalizatoin

There are many future plans for this repository:
- Integration with pyfissa (for onsite finite-size scaling analysis). Right now I'm doing the data analysis in spreadsheets, I know- it's embarassing.
- MCMC implementations for other systems (O(N), N-spins Ising)
- Implementations of other existing common RG procedures
- Generalization of RSMI algorithm.
  - One-hot neurons: to extend this to the above
  - General lattices: (next would be the triangular lattice)
  - Harmonium-family: consider more complicated energy functions than RBM's linear function.
  - Other approximations to the mutual information:
    - More terms in the Koch-Janusz and Ringel's cumulant expansion. Ignore this altogether
    - Oord et al. (https://arxiv.org/pdf/1807.03748.pdf?fbclid=IwAR2G_jEkb54YSIvN0uY7JbW9kfhogUq9KhKrmHuXPi34KYOE8L5LD1RGPTo)
    - Hjelm et al. (https://arxiv.org/pdf/1808.06670.pdf?fbclid=IwAR2WxWc4eR_fo3tV-vUxElKbqKxNWAapGxRbvyQhtum7os3ACSISqb0D1xw)
- Momentum-space mutual information algorithm
  - QFTs: otherwise use ODENets
