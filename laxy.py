import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam

class OPT():
  def __init__(self, fn, params, lr=1e-3):
    self.k = 0
    self.opt_init, self.opt_update, self.opt_params = adam(step_size=lr)
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

def MRF(params=None):
  '''markov random field'''
  def init_params(L, A, key=None):
    return {"w":jnp.zeros((L,A,L,A)),
            "b":jnp.zeros((L,A))}
  def layer(x, use_bias=True, l2=False):
    b,w = params["b"],params["w"]
    L,A = b.shape
    w = w * (1-jnp.eye(L)[:,None,:,None])
    w = 0.5 * (w + w.transpose([2,3,0,1]))
    x_pred = jnp.tensordot(x,w,2) 
    if use_bias: x_pred += b
    if l2:
      l2_loss = 0.5 * (L-1) * A * jnp.square(w).sum() + jnp.square(b).sum()
      return x_pred, l2_loss
    else:
      return x_pred

  if params is None: return init_params
  else: return layer

def CNN(params=None):
  '''convolution'''
  def init_params(in_dims, out_dims, win, key):
    return {"w":jax.nn.initializers.glorot_normal()(key,(out_dims,in_dims,win)),
            "b":jnp.zeros(out_dims)}  
  def layer(x, use_bias=True, stride=1, padding="SAME", key=None):
    w = params["w"]
    x = x.transpose([0,2,1])
    y = jax.lax.conv(x,w,(stride,),padding=padding)
    y = y.transpose([0,2,1]) 
    if use_bias: y += params["b"]
    return y
  
  if params is None: return init_params
  else: return layer

def CNN_2D(params=None):
  '''2D convolution'''
  def init_params(in_dims, out_dims, win, key):
    return {"w":jax.nn.initializers.glorot_normal()(key,(out_dims,in_dims,win,win)),
            "b":jnp.zeros(out_dims)}
  def layer(x, use_bias=True, stride=1, padding="SAME"):
    x = x.transpose([0,3,1,2]) # (batch, channels, row, col)
    y = jax.lax.conv(x,params["w"],(stride,stride),padding=padding) # (batch, filters, row, col)
    y = y.transpose([0,2,3,1]) # (batch, row, col, filters)
    if use_bias: y += params["b"]
    return y
  if params is None: return init_params
  else: return layer

def DENSE(params=None):
  '''dense or linear layer'''
  def init_params(in_dims, out_dims, key):
    return {"w":jax.nn.initializers.glorot_normal()(key,(in_dims,out_dims)),
            "b":jnp.zeros(out_dims)}
  def layer(x, use_bias=True):
    y = x @ params["w"]
    if use_bias: y += params["b"]
    return y
  if params is None: return init_params
  else: return layer

def GRU(params=None):
  '''gated recurrent unit'''
  # wikipedia.org/wiki/Gated_recurrent_unit
  def init_params(in_dims, out_dims, key):
    gn = lambda k,i: jax.nn.initializers.glorot_normal()(k,(i,out_dims))
    zr = lambda i: jnp.zeros(i)
    k = jax.random.split(key, num=6)
    return {"z":{"w":gn(k[0],in_dims),"u":gn(k[1],out_dims),"b":zr(out_dims)},
            "r":{"w":gn(k[2],in_dims),"u":gn(k[3],out_dims),"b":zr(out_dims)},
            "h":{"w":gn(k[4],in_dims),"u":gn(k[5],out_dims),"b":zr(out_dims)}}

  def layer(x, h0=None):
    def gru_cell(h,x):
      zt = jax.nn.sigmoid(x@params["z"]["w"] + h@params["z"]["u"] + params["z"]["b"])
      rt = jax.nn.sigmoid(x@params["r"]["w"] + h@params["r"]["u"] + params["r"]["b"])
      ht = jnp.tanh(x@params["h"]["w"] + (h*rt)@params["h"]["u"] + params["h"]["b"])      
      h = (1-zt)*h + zt*ht
      return h,h

    if h0 is None: h0 = jnp.zeros([x.shape[0],params["z"]["w"].shape[1]])
    h,seq = jax.lax.scan(gru_cell,h0,x.swapaxes(0,1))
    return seq.swapaxes(0,1)

  if params is None: return init_params
  else: return layer
