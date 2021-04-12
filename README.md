# laxy
This is my "lazy" wrapper around jax, because I'm too lazy to explicitly write it out each time (set the key/seed for random number generator, setting up the optimizer, keeping track of weights).

Finally implementations of MRF, GRU, LSTM layers are provided. Since these are not currently part of stax ( jax's experimental NN library).

Examples:
* [linear regression and stax integration](https://colab.research.google.com/github/sokrypton/laxy/blob/main/laxy_example.ipynb)
* [gremlin](https://colab.research.google.com/github/sokrypton/laxy/blob/main/gremlin_jax.ipynb)
