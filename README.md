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

# Usage Notes
Project is entirely python. Requires numpy, and pandas to work.
Matplotlib is used for plotting sometimes.

# For Future Users

Here is a very brief rundown of how the alg works:

Basically, initialize Master by passing it a non-systematic G, and the raw data as an np.ndarray which needs to be padded
correctly beforehand (potential update would be to automate this... would be super easy). Be aware the right now the Master class
can make a decoder automatically, but only for a radius 2 code... Then to query master, pass it a quantized w vector 
which is 1xn or dx1, and the master will figure out if it is supposed to be a left or right multiplication. An "index"
also needs to be passed to create the lookup table for making +-1 queries. The Master instance will return the product of
the multiplication.

For gradient based computations, there is a gradient descent alg which should work with any loss function. It is a good idea
to use a standard scaler on the data beforehand in order to avoid overflow errors which will break the computations.

The run_test function is what we have been using to record data about the algorithm's performance. It will use a dictionary
to compile a csv file with access measurements, and also record the loss and stuff at the end of each alg. Examples of how
to use the file are also in the notebook.

That's all we have to say for now!