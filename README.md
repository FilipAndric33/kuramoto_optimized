# kuramoto
This is the optimized version of the github repo available at: https://github.com/fabridamicelli/kuramoto.

## Usage
Run sim.py to run the simulation. This version is optimized via Numpy and Cython for CPU-bound tasks and PyTorch for GPU-bound tasks.
The dask_simulation.py simulation is optimized for running multiple instances of the simulation concurrently with different levels of coupling. The main parameter that needs to be adjusted for your specific use case is the number of nodes (n_nodes).

## Results

Results achieved by the optimizations implemented are best represented via graphs.

![png](https://github.com/FilipAndric33/kuramoto_optimized/tree/master/plots/01_full_model_run.png)

![png](https://github.com/FilipAndric33/kuramoto_optimized/tree/master/plots/02_phase_coherence.png)

![png](https://github.com/FilipAndric33/kuramoto_optimized/tree/master/plots/03_mean_frequency.png)

![png](https://github.com/FilipAndric33/kuramoto_optimized/tree/master/plots/04_dask_unoptimized.png)

![png](https://github.com/FilipAndric33/kuramoto_optimized/tree/master/plots/07_speedup_overview.png)

## Kuramoto model 101
- The [Kuramoto model](https://en.wikipedia.org/wiki/Kuramoto_model) is used to study a wide range of systems with synchronization behaviour.
- It is a system of _N_ coupled periodic oscillators.
- Each oscillator has its own _natural frequency_ _omega<sub>i_, i.e., constant angular velocity. 
- Usually, the distribution of natural frequencies is choosen to be a gaussian-like symmetric function.
- A random initial (angular) position _theta<sub>i_ is assigned to each oscillator.
- The oscillator's state (position) _theta<sub>i_ is governed by the following differential equation:

![jpg](https://github.com/fabridamicelli/kuramoto_model/blob/master/images/derivative.jpg)
      

where K is the coupling parameter and _M<sub>i_ is the number of oscillators interacting with oscillator _i_. 
_A_ is the _adjacency matrix_ enconding the interactions - typically binary and undirected (symmetric), such that if node _i_ interacts with node _j_, _A<sub>ij_ = 1, otherwise 0.
The basic idea is that, given two oscillators, the one running ahead is encouraged to slow down while the one running behind to accelerate.

In particular, the classical set up has _M = N_, since the interactions are all-to-all (i.e., a complete graph). Otherwise, _M<sub>i_ is the degree of node _i_.

## Kuramoto model 201
A couple of facts in order to gain intuition about the model's behaviour:
- If synchronization occurs, it happens abruptly.
- That is, synchronization might not occur.
- Partial synchronization is a possible outcome.
- The order parameter _R<sub>t_ measures global synchronization at time _t_. It is basically the normalized length of the sum of all vectors (oscillators in the complex plane).
- About the global order parameter _R<sub>t_:
  - constant, in the double limit N -> inf, t -> inf
  - independent of the initial conditions
  - depends on coupling strength
  - it shows a sharp phase transition (as function of coupling)
- Steady solutions can be computed assuming _R<sub>t_ constant. The result is basically that each oscillator responds to the mean field produced by the rest.
- In the all-to-all connected scenaria, the critical coupling _K<sub>c_ can be analytically computed and it depends on the spread of the natural frequencies distribution (see English, 2008)
- The higher the dimension of the lattice on which the oscillators are embedded, the easier it is to synchronize. For example, there isn't any good synchronization in one dimension, even with strong coupling. In two dimensions it is not clear yet. From 3 dimensions on, the model starts behaving more like the mean field prediction.

For more and better details, [this talk](https://www.youtube.com/watch?v=5zFDMyQ8z8g) by the great Steven Strogatz is a nice primer. Also, visiting the original repo.

## Tests
Run tests with
```bash
make test
```

## Citing

If you find this package useful for a publication, then please use the following BibTeX to cite it:
```
@misc{kuramoto,
  author = {Damicelli, Fabrizio},
  title = {Python implementation of the Kuramoto model},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fabridamicelli/kuramoto}},
}
```

## References & links 
- [English, 2008](http://doi.org/10.1088/0143-0807/29/1/015). Synchronization of oscillators: an ideal introduction to phase transitions.
- [Dirk Brockmann's explorable](http://www.complexity-explorables.org/explorables/kuramoto/). “Ride my Kuramotocycle!”.
- [Math Insight - applet](https://mathinsight.org/applet/kuramoto_order_parameters). The Kuramoto order parameters.
- [Kuramoto, Y. (1984)](http://doi.org/10.1007/978-3-642-69689-3). Chemical Oscillations, Waves, and Turbulence.
- [Steven Strogatz - talk](https://www.youtube.com/watch?v=5zFDMyQ8z8g). Coupled Oscillators That Synchronize Themselves
