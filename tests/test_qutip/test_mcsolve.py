import pytest
import jax
import jax.numpy as jnp
import qutip as qt
import qutip_jax as qjax
from qutip import mcsolve
from functools import partial

# Use JAX backend for QuTiP
qjax.use_jax_backend()

# Define time-dependent functions
@partial(jax.jit, static_argnames=("omega",))
def H_1_coeff(t, omega):
    return 2.0 * jnp.pi * 0.25 * jnp.cos(2.0 * omega * t)

# Test setup for gradient calculation
def setup_system(size=2):
    a = qt.destroy(size).to("jax")  
    sm = qt.sigmax().to("jax")  

    # Define the Hamiltonian
    H_0 = 2.0 * jnp.pi * a.dag() * a + 2.0 * jnp.pi * sm.dag() * sm
    H_1_op = sm * a.dag() + sm.dag() * a

    H = [H_0, [H_1_op, qt.coefficient(H_1_coeff, args={"omega": 1.0})]]

    state = qt.basis(size, size-1).to("jax")

    # Define collapse operators and observables
    c_ops = [jnp.sqrt(0.1) * a]
    e_ops = [a.dag() * a, sm.dag() * sm]

    # Time list
    tlist = jnp.linspace(0.0, 10.0, 101)
    
    return H, state, tlist, c_ops, e_ops

# Function for which we want to compute the gradient
def f(omega, H, state, tlist, c_ops, e_ops):
    H[1][1] = qt.coefficient(H_1_coeff, args={"omega": omega})
    
    result = mcsolve(H, state, tlist, c_ops, e_ops, ntraj=10, options={"method": "diffrax"})
    
    return result.expect[0][-1].real

# Pytest test case for gradient computation
@pytest.mark.parametrize("omega_val", [1.0, 2.0, 3.0])
def test_gradient_mcsolve(omega_val):
    H, state, tlist, c_ops, e_ops = setup_system(size=2)
    
    # Compute the gradient with respect to omega
    grad_func = jax.grad(lambda omega: f(omega, H, state, tlist, c_ops, e_ops))
    gradient = grad_func(omega_val)
    
    # Check if the gradient is not None and has the correct shape
    assert gradient is not None
    assert gradient.shape == ()  
    assert jnp.isfinite(gradient)  
