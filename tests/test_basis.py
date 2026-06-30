import numpy as np

from kanop.basis import hermite_basis, laguerre_basis, laguerre_total_degree_cross_basis, weighted_laguerre_basis


def test_laguerre_shape():
    x = np.array([1.0, 2.0, 3.0])
    phi = laguerre_basis(x, max_order=5)
    assert phi.shape == (3, 6)
    assert np.allclose(phi[:, 0], 1.0)


def test_weighted_laguerre_shape():
    x = np.array([1.0, 2.0, 3.0])
    phi = weighted_laguerre_basis(x, max_order=5)
    assert phi.shape == (3, 6)


def test_hermite_shape():
    x = np.array([1.0, 2.0, 3.0])
    phi = hermite_basis(x, max_order=5)
    assert phi.shape == (3, 6)


def test_laguerre_2d_cross_basis_has_15_terms():
    x = np.array([[100.0, 100.0], [101.0, 100.5]])
    phi = laguerre_total_degree_cross_basis(x, max_total_order=4)
    assert phi.shape == (2, 15)
