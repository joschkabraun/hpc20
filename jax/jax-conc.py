import jax
import jax.numpy as jnp
import concurrent.futures
import numpy as onp
from functools import partial

#@jax.jit
def f(x):
    for _ in range(10):
        y = jax.device_put(x)
        x = jax.device_get(y)
    return x

#@jax.jit
def g(x):
    return x - 3.

xs = [onp.random.randn(i) for i in range(10)]
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(partial(g, x)) for x in xs]
    ys = [f.result() for f in futures]
for x, y in zip(xs, ys):
    if not all(jnp.equal(x,y)):
        print("These are not the same")
        print(x)
        print(y)
        print("----------------------")
print("done")
