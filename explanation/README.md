# Algorithm explanations

Start with [The winning big-cell algorithm, end to end](winning_bigcell_algorithm.md).

It explains the production path in simple language, including:

- what broad culling means;
- every CUDA kernel launch;
- grid and block shapes;
- inputs and outputs of each kernel;
- shared-memory layout;
- the home-cell rule;
- raw-AABB reuse;
- the 15 closest-feature tests;
- a complete worked trace for one triangle pair.
