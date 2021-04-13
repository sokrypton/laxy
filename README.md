# laxy
This is my "lazy" wrapper around jax, intended to minimize extra work in setting up optimization for simple custom models. For more advanced deep-nn models, I'd recommend libraries like [Haiku](https://github.com/deepmind/dm-haiku), [Elegy](https://github.com/poets-ai/elegy), [Flax](https://github.com/google/flax), or [Trax](https://github.com/google/trax).

#### Philosophy: "Optimizing a simple model shouldn't require more than 2 lines of code"
```python
import laxy
import jax.numpy as jnp

def model(params, inputs):
  out = inputs["x"] * params["m"] + params["b"]
  loss = jnp.square(inputs["y"] - out)
  return out, loss

opt = laxy.OPT(model, params={"a":1.0,"b":0.0})
opt.fit(inputs={"x":x,"y":y})
```

Examples:
* [linear regression](https://colab.research.google.com/github/sokrypton/laxy/blob/main/laxy_example.ipynb)
* [gremlin](https://colab.research.google.com/github/sokrypton/laxy/blob/main/gremlin_jax.ipynb)
