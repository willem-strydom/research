# Purpose
This project was developed to implement and test a low access query system from a 
simulated distributed-coded storage system. This system is then implemented in a quantized
logistic regression algorithm. The project is based on this [paper](https://arxiv.org/abs/2305.06101).

# Contributions
This project is supervised and guided by:

- **Vinayak Ramkumar**
  *Postdoctoral Research Fellow at Tel Aviv University*
- **Netanel Raviv**
  *Assistant Professor of Computer Science at Washington University in St. Louis*  
  [Lab](https://sites.wustl.edu/ravivlab/)


This Repository was developed by:

- **Willem Strydom**
- **Jinzhao Kang**

# Usage notes
Project is entirely python. Requires numpy, and pandas to work.
Matplotlib is used for plotting sometimes.

# Problems

- The quantization is likely not optimal... there honestly isn't that much that can be done about that we imagine.
- Creating the lookup table for decoding is O(2^n), which probably prevents low access ratio codes from being used.
- Convergence is noisy and bad at low quantization levels.

# Next Up

- make the quantization dynamic. The algorithm would adjsut the quantization rate, maybe just monotonically, based on
some criterion until the point where there are no more access gains, and the just run as an unquantized version of the algorithm.