import pytest
import jax.numpy as jnp
from jax import jit, grad
from qutip import Qobj, basis, rand_dm, sigmax, identity, tensor, expect
import qutip.settings
import qutip_jax

# Set JAX backend for QuTiP
qutip.settings.core["auto_real_casting"] = False
qutip_jax.use_jax_backend()
tol = 1e-6  # Tolerance for assertion

# Initialize quantum objects for testing
with qutip.CoreOptions(default_dtype="jax"):
    rho = rand_dm(2)
    ket = basis(2, 0)
    bra = ket.dag()
    op1 = sigmax()
    identity_op = identity(2)
    composite_op = tensor(op1, identity_op)

def expectation_value(op: Qobj, state: Qobj) -> float:
    """
    Compute the expectation value of an operator with respect to a quantum state.

    Args:
        op (Qobj): The operator (as a Qobj).
        state (Qobj): The quantum state (as a Qobj).

    Returns:
        float: The expectation value.
    """
    return expect(op, state)

# Test case for Qobj functions with jax.jit
@pytest.mark.parametrize("func_name, func", [
    ("copy", lambda x: x.copy()),
    ("conj", lambda x: x.conj()),
    ("contract", lambda x: x.contract()),
    ("cosm", lambda x: x.cosm()),
    ("dag", lambda x: x.dag()),
    ("eigenenergies", lambda x: x.eigenenergies()),
    ("expm", lambda x: x.expm()),
    ("inv", lambda x: x.inv()),
    ("logm", lambda x: x.logm()),
    ("matrix_element", lambda x: x.matrix_element(ket, ket)),
    ("norm", lambda x: x.norm()),
    ("overlap", lambda x: x.overlap(op1)),
    ("ptrace", lambda x: x.ptrace([0])),
    ("purity", lambda x: x.purity()),
    ("sinm", lambda x: x.sinm()),
    ("sqrtm", lambda x: x.sqrtm()),
    ("tr", lambda x: x.tr()),
    ("trans", lambda x: x.trans()),
    ("transform", lambda x: x.transform(identity_op)),
    ("unit", lambda x: x.unit())
])
def test_qobj_jit(func_name, func):
    # Create a jitted function using the given Qobj function
    def jit_func(op):
        return func(op)

    # Apply jit to the function
    func_jit = jit(jit_func)
    result_jit = func_jit(op1)

    # Check if jit result is not None
    assert result_jit is not None
    print(f"JIT result of {func_name} with respect to Qobj data:", result_jit)

# Test case for Qobj functions with jax.grad
@pytest.mark.parametrize("func_name, func", [
    ("conj", lambda x: x.conj()),
    ("contract", lambda x: x.contract()),
    ("cosm", lambda x: x.cosm()),
    ("dag", lambda x: x.dag()),
    ("eigenenergies", lambda x: x.eigenenergies()),
    ("expm", lambda x: x.expm()),
    ("inv", lambda x: x.inv()),
    ("overlap", lambda x: x.overlap(op1)),
    ("purity", lambda x: x.purity()),
    ("sinm", lambda x: x.sinm()),
    ("tr", lambda x: x.tr()),
])
def test_qobj_grad(func_name, func):
    # Create a differentiable function using the given Qobj function
    def grad_func(op1):
        return jnp.real(func(op1))

    # Apply grad to the function
    grad_func = grad(grad_func)
    grad_result = grad_func(op1)

    # Check if the gradient is not None
    assert grad_result is not None