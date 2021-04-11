import jax
import jax.numpy as jnp
import random
from jax.experimental.optimizers import adam

def get_random_key(seed=None):
  if seed is None: seed = random.randint(0,2147483647)
  return jax.random.PRNGKey(seed) 
  
class KEY():
  '''generate random key'''
  def __init__(self, seed=None):
    self.key = get_random_key(seed)    
  def get(self, num=1):
    if num > 1:
      self.key, *subkeys = jax.random.split(self.key, num=(num+1))
      return subkeys
    else:
      self.key, subkey = jax.random.split(self.key)
      return subkey

class OPT():
  def __init__(self, fn, params, lr=1e-3, optimizer=adam):
    self.k = 0
    self.opt_init, self.opt_update, self.opt_params = optimizer(step_size=lr)
    self.opt_state = self.opt_init(params) 
    self.fn = jax.jit(fn)
    self.d_fn = jax.value_and_grad(self.fn)

    def update(k, state, inputs):      
      loss, grad = self.d_fn(self.opt_params(state), inputs)
      state = self.opt_update(k, grad, state)
      return state,loss
    self.update = jax.jit(update)

  def train_on_batch(self, inputs):
    self.opt_state,loss = self.update(self.k, self.opt_state, inputs)
    self.k += 1
    return loss

  def set_params(self, params):
    self.opt_state = self.opt_init(params)
    
  def get_params(self):
    return self.opt_params(self.opt_state)
  
  def get_loss(self, inputs):
    return self.fn(self.get_params(), inputs)
  
#################
# LAYERS
#################

def STAX(stax_layers, input_shape, key=None, seed=None):
  if key is None: key = get_random_key(seed)
  _init_params, _layer = stax_layers
  _params = _init_params(key, input_shape)[1]
  return _params, _layer

def MRF(params=None):
  '''markov random field'''
  def init_params(L, A, use_bias=True, key=None, seed=None):
    params = {"w":jnp.zeros((L,A,L,A))}
    if use_bias: params["b"] = jnp.zeros((L,A))
    return params
  
  def layer(x, l2=False):
    w = params["w"]
    L,A = w.shape[:2]
    w = w * (1-jnp.eye(L)[:,None,:,None])
    w = 0.5 * (w + w.transpose([2,3,0,1]))
    y = jnp.tensordot(x,w,2) 
    if "b" in params: y += params["b"]
      
    if l2:
      l2_loss = 0.5 * (L-1) * A * jnp.square(w).sum() 
      if "b" in params: l2_loss += jnp.square(params["b"]).sum()
      return y, l2_loss
    else:
      return y

  if params is None: return init_params
  else: return layer

def Conv1D(params=None):
  '''1D convolution'''
  def init_params(in_dims, out_dims, win, use_bias=True, key=None, seed=None):
    if key is None: key = get_random_key(seed)
    params = {"w":jax.nn.initializers.glorot_normal()(key,(out_dims,in_dims,win))}
    if use_bias: params["b"] = jnp.zeros(out_dims)
    return params
      
  def layer(x, stride=1, padding="SAME"):
    x = x.transpose([0,2,1])
    y = jax.lax.conv(x,params["w"],(stride,),padding=padding)
    y = y.transpose([0,2,1]) 
    if "b" in params: y += params["b"]
    return y
  
  if params is None: return init_params
  else: return layer

def Conv2D(params=None):
  '''2D convolution'''
  def init_params(in_dims, out_dims, win, use_bias=True, key=None, seed=None):
    if key is None: key = get_random_key(seed)
    params = {"w":jax.nn.initializers.glorot_normal()(key,(out_dims,in_dims,win,win))}
    if use_bias: params["b"] = jnp.zeros(out_dims)
    return params
      
  def layer(x, use_bias=True, stride=1, padding="SAME"):
    x = x.transpose([0,3,1,2]) # (batch, channels, row, col)
    y = jax.lax.conv(x,params["w"],(stride,stride),padding=padding) # (batch, filters, row, col)
    y = y.transpose([0,2,3,1]) # (batch, row, col, filters)
    if use_bias: y += params["b"]
    return y
  
  if params is None: return init_params
  else: return layer

def Dense(params=None):
  '''dense or linear layer'''
  def init_params(in_dims, out_dims, use_bias=True, key=None, seed=None):
    if key is None: key = get_random_key(seed)
    params = {"w":jax.nn.initializers.glorot_normal()(key,(in_dims,out_dims))}
    if use_bias: params["b"] = jnp.zeros(out_dims)
    return params
  
  def layer(x, use_bias=True):
    y = x @ params["w"]
    if "b" in params: y += params["b"]
    return y
  
  if params is None: return init_params
  else: return layer

def GRU(params=None):
  '''Gated recurrent unit (GRU)'''
  # wikipedia.org/wiki/Gated_recurrent_unit
  def init_params(in_dims, out_dims, key=None, seed=None):
    if key is None: key = get_random_key(seed)
    G = lambda k,i: jax.nn.initializers.glorot_normal()(k,(i,out_dims))
    k = jax.random.split(key, num=6)
    return {"z":{"w":G(k[0],in_dims),"u":G(k[1],out_dims),"b":jnp.zeros(out_dims)},
            "r":{"w":G(k[2],in_dims),"u":G(k[3],out_dims),"b":jnp.zeros(out_dims)},
            "h":{"w":G(k[4],in_dims),"u":G(k[5],out_dims),"b":jnp.zeros(out_dims)}}

  def layer(x):
    def gru_cell(h,x):
      zt = jax.nn.sigmoid(x@params["z"]["w"] + h@params["z"]["u"] + params["z"]["b"])
      rt = jax.nn.sigmoid(x@params["r"]["w"] + h@params["r"]["u"] + params["r"]["b"])
      ht = jnp.tanh(x@params["h"]["w"] + (h*rt)@params["h"]["u"] + params["h"]["b"])      
      h = (1-zt)*h + zt*ht
      return h,h

    out_dims = params["z"]["w"].shape[1]
    h0 = jnp.zeros(out_dims)
    h,seq = jax.lax.scan(gru_cell,h0,x)
    return seq

  if params is None: return init_params
  else: return jax.vmap(layer)

def LSTM(params=None):
  '''Long short-term memory (LSTM)'''
  # wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
  def init_params(in_dims, out_dims, key=None, seed=None):
    if key is None: key = get_random_key(seed)
    G = lambda k,i: jax.nn.initializers.glorot_normal()(k,(i,out_dims))
    k = jax.random.split(key, num=8)
    return {"f":{"w":G(k[0],in_dims),"u":G(k[1],out_dims),"b":jnp.ones(out_dims)},  # forget gate
            "i":{"w":G(k[2],in_dims),"u":G(k[3],out_dims),"b":jnp.zeros(out_dims)}, # input gate
            "o":{"w":G(k[4],in_dims),"u":G(k[5],out_dims),"b":jnp.zeros(out_dims)}, # output gate
            "g":{"w":G(k[6],in_dims),"u":G(k[7],out_dims),"b":jnp.zeros(out_dims)}} # change gate

  def layer(x):
    def lstm_cell(hc,x):
      h,c = hc
      f_t = jax.nn.sigmoid(x@params["f"]["w"] + h@params["f"]["u"] + params["f"]["b"])
      i_t = jax.nn.sigmoid(x@params["i"]["w"] + h@params["i"]["u"] + params["i"]["b"])
      o_t = jax.nn.sigmoid(x@params["o"]["w"] + h@params["o"]["u"] + params["o"]["b"])
      g_t = jnp.tanh(x@params["g"]["w"] + h@params["g"]["u"] + params["g"]["b"])
      c_t = f_t * c + i_t * g_t
      h_t = o_t * jnp.tanh(c_t)
      return (h_t,c_t),c_t

    out_dims = params["f"]["w"].shape[1]
    h0 = jnp.zeros(out_dims)
    h,seq = jax.lax.scan(lstm_cell,(h0,h0),x)
    return seq

  if params is None: return init_params
  else: return jax.vmap(layer)
