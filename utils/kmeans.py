import jax
import jax.numpy as jnp

def k_means(X, X_weight, n_clusters=8, n_init=10, max_iter=300, tol=1e-4, seed=0):
  '''kmeans implemented in jax'''
  def _dist(a,b):
    sm = a @ b.T
    a_norm = jnp.square(a).sum(-1)
    b_norm = jnp.square(b).sum(-1)
    return jnp.abs(a_norm[:,None] + b_norm[None,:] - 2 * sm)

  def _kmeans_plus_plus(key):
    n_samples, n_features = X.shape
    def loop(m,c):
      n,k = c
      inf_mask = jnp.inf * (jnp.arange(n_clusters) > n)
      p = (inf_mask + _dist(X,m)).min(-1)
      i = jax.random.choice(k,jnp.arange(n_samples),p=p/p.sum())
      return m.at[n].set(X[i]), None

    i = jax.random.choice(key,jnp.arange(n_samples))
    init_means = jnp.zeros((n_clusters,n_features)).at[0].set(X[i])
    carry = (jnp.arange(1,n_clusters), jax.random.split(key,n_clusters-1))
    return jax.lax.scan(loop, init_means, carry)[0]

  def E(means):
    # get labels
    return jax.nn.one_hot(_dist(X,means).argmin(-1), n_clusters)

  def M(labels):
    # get means
    labels = labels * X_weight[:,None]
    labels /= labels.sum(0) + 1e-8
    return labels.T @ X
  
  def sco(means):
    # compute score: sum(min(dist(X,means)))
    inertia = _dist(X,means).min(-1)
    return (X_weight * inertia).sum()

  def single_run(key):
    # initialize
    init_means = _kmeans_plus_plus(key)

    # run EM
    def update(x):
      mu = M(E(x[0]))
      return mu, sco(mu), x[1]
    init = update((init_means,jnp.inf,None))
    means = jax.lax.while_loop(lambda x:(x[2]-x[1])>tol, update, init)[0]
    return {"means":means,"labels":E(means),"sco":sco(means)}

  key = jax.random.PRNGKey(seed)

  # mulitple runs
  out = jax.vmap(single_run)(jax.random.split(key,n_init))
  i = out["sco"].argmin()
  out = {k:v[i] for k,v in out.items()}

  cat = (out["labels"] * X_weight[:,None]).sum(0) / X_weight.sum()
  return {**out, "cat":cat}
