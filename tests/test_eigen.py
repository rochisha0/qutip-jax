import qutip
import qutip_jax
import pytest
import jax.numpy as jnp
import numpy as np



def is_eigen_set(oper, vals, vecs):
    for val, vec in zip(vals, vecs.T):
        assert abs(jnp.linalg.norm(vec) - 1) < 1e-13
        assert abs(vec.T.conj() @ oper @ vec - val) < 1e-13


def test_eigen_known_oper():
    mat = qutip.num(10, dtype="jax").data
    spvals, spvecs = qutip_jax.eigs_jaxarray(mat)
    expected = np.arange(10)
    is_eigen_set(mat._jxa, spvals, spvecs._jxa)
    np.testing.assert_allclose(spvals, expected, atol=1e-13)


@pytest.mark.parametrize(["rand"], [
    pytest.param(qutip.rand_herm, id="hermitian"),
    pytest.param(qutip.rand_unitary, id="non-hermitian"),
])
@pytest.mark.parametrize("order", ['low', 'high'])
def test_eigen_rand_oper(rand, order):
    mat = rand(10, dtype="jax").data
    isherm = rand is qutip.rand_herm
    kw = {"isherm": isherm, "sort": order}
    spvals, spvecs = qutip_jax.eigs_jaxarray(mat, vecs=True, **kw)
    sp_energies = qutip_jax.eigs_jaxarray(mat, vecs=False, **kw)
    if order == 'low':
        assert np.all(np.diff(spvals).real >= 0)
    else:
        assert np.all(np.diff(spvals).real <= 0)
    is_eigen_set(mat._jxa, spvals, spvecs._jxa)
    np.testing.assert_allclose(spvals, sp_energies, atol=5e-15)


@pytest.mark.parametrize("rand", [
    pytest.param(qutip.rand_herm, id="hermitian"),
    pytest.param(qutip.rand_unitary, id="non-hermitian"),
])
@pytest.mark.parametrize("order", ['low', 'high'])
@pytest.mark.parametrize("N", [1, 5, 8, 9])
def test_FewState(rand, order, N):
    mat = rand(10, dtype="jax").data
    isherm = rand is qutip.rand_herm
    kw = {"isherm": isherm, "sort": order}
    spvals, spvecs = qutip_jax.eigs_jaxarray(mat, vecs=True, eigvals=N, **kw)
    sp_energies = qutip_jax.eigs_jaxarray(mat, vecs=False, eigvals=N, **kw)
    all_spvals = qutip_jax.eigs_jaxarray(mat, vecs=False, **kw)
    assert np.allclose(all_spvals[:N], spvals)
    assert np.allclose(all_spvals[:N], sp_energies)
    is_eigen_set(mat._jxa, spvals, spvecs._jxa)
    if order == 'low':
        assert np.all(np.diff(spvals).real >= 0)
    else:
        assert np.all(np.diff(spvals).real <= 0)
