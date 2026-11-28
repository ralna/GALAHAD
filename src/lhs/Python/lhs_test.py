from galahad import lhs
import numpy as np
import pytest


# lhs.ihs computes n_points Latin hypercube samples in n_dimen dimensions:
# each coordinate takes the values 1, ..., n_points, and no two points share
# a coordinate value, so each column of X is a permutation of 1, ..., n_points

@pytest.fixture
def terminate():
    yield
    lhs.terminate()

def ihs(n_dimen, n_points, seed):
    options, inform = lhs.initialize()
    options['print_level'] = 0
    return lhs.ihs(n_dimen, n_points, seed, options)


@pytest.mark.parametrize('n_dimen, n_points', [(2, 5), (3, 10), (1, 4),
                                               (4, 1)])
def test_lhs_latin(terminate, n_dimen, n_points):
    X, seed = ihs(n_dimen, n_points, 1)
    assert lhs.information()['status'] == 0
    assert X.shape == (n_points, n_dimen)
    for j in range(n_dimen):
        assert sorted(X[:, j]) == list(range(1, n_points + 1))

def test_lhs_repeatable(terminate):
    # the same seed gives the same samples
    X1, seed1 = ihs(3, 10, 7)
    lhs.terminate()
    X2, seed2 = ihs(3, 10, 7)
    assert np.array_equal(X1, X2)
    assert seed1 == seed2

def test_lhs_seed_updated(terminate):
    # the returned seed may be used for further samples, which still form
    # a Latin hypercube
    X1, seed = ihs(3, 10, 7)
    X2, seed = lhs.ihs(3, 10, seed)
    assert lhs.information()['status'] == 0
    for j in range(3):
        assert sorted(X2[:, j]) == list(range(1, 11))

def test_lhs_get_seed():
    seed = lhs.get_seed()
    assert isinstance(seed, int)
