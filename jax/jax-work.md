
By reading examples it appears as that many numpy methods of Jax are coded in jax/numpy/lax_numpy.py
So implementing new functions seems appropriate.
They have a @_wraps(onp.method_name) where onp is the original numpy. What does that mean? _wraps(fun, update_doc=True, lax_description="") is a method that wraps to the numpy method.

Methods which I can implement:
ediff1d -> not dependend on unique; but ravel, asanarray, all, empty, __array_wrap__, subtract
ravel is implemented for order='C' -> as needed
asanarray -> not implemented, but Oliver argues it is the same as asarray?
all -> test if all array el.s of given axis evaluate to True
empty -> can't create uninitialized arrays; but use zeros for that
__array_wrap__ -> look more under the curtain !!!!! by olivers comment, there is no subclassing, i.e., not needed
subtract -> is not in list of unimplemented funcs, so should be implemented

in1d -> depends on unique, ravel
intersect1d -> depends on unique, ravel, argsort, asanarray, concatenate
isin -> depends on in1d, reshape, asarray
setdiff1d -> depends on asarray, ravel, unique
setxor1d -> depends on unique, concatenate
union1d -> depends on unique, concatentate
