import jax.numpy as jnp
import numpy as np
from qutip import settings

__all__ = ["use_jax_backend"]

def use_jax_backend():
    settings.core['numpy_backend'] = jnp
    settings.core["default_dtype"] = "jax"

def use_numpy_backend():
    settings.core['numpy_backend'] = np
    settings.core["default_dtype"] = None

