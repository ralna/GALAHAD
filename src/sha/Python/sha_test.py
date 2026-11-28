from galahad import sha
import numpy as np
import pytest


# sha estimates the entries of a sparse symmetric matrix H, given its
# sparsity pattern (the upper triangle, by co-ordinates) and m pairs of
# steps s^(k) and differences y^(k) = H s^(k), stored as the rows of strans
# and ytrans. analyse_matrix returns the number of pairs the chosen
# algorithm requires; with that many (or more) generic pairs, H is
# recovered exactly. order gives the rows of strans and ytrans in their
# preferred order (indices from 0)

# 1) as in test_sha.py: an "arrow" matrix with a full first row and column
# and a diagonal, with the values 1, ..., 9
n = 5
row = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4])
col = np.array([0, 1, 2, 3, 4, 1, 2, 3, 4])
ne = len(row)
val_h = np.arange(1.0, ne + 1.0)
H = np.zeros((n, n))
H[row, col] = val_h
H[col, row] = val_h

# 2) a tridiagonal matrix
row_t = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4])
col_t = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4])
val_t = np.array([4.0, -1.0, 4.0, -1.0, 4.0, -1.0, 4.0, -1.0, 4.0])
H_t = np.zeros((n, n))
H_t[row_t, col_t] = val_t
H_t[col_t, row_t] = val_t

ALGORITHMS = [1, 2, 3, 4, 5]

@pytest.fixture
def terminate():
    yield
    sha.terminate()

def analyse(algorithm, row=row, col=col):
    o = sha.initialize()
    o['print_level'] = 0
    o['approximation_algorithm'] = algorithm
    return sha.analyse_matrix(n, len(row), row, col, o)

def differences(H, m, rows=None):
    # m generic steps (rows of strans) and the differences H s (rows of
    # ytrans), in arrays with rows (>= m) rows
    rows = rows or m
    rng = np.random.default_rng(1)
    strans = np.zeros((rows, n))
    strans[:m] = rng.uniform(-1.0, 1.0, (m, n))
    ytrans = strans @ H
    return strans, ytrans

def recover(m, strans, ytrans, ne=ne, **kw):
    ls1, ls2 = strans.shape
    ly1, ly2 = ytrans.shape
    return sha.recover_matrix(ne, m, ls1, ls2, strans, ly1, ly2, ytrans,
                              **kw)


# the number of pairs required for the arrow matrix: algorithm 2 exploits
# symmetry, the others need as many as the largest number of entries in a
# row
REQUIRED = {1: 5, 2: 2, 3: 5, 4: 5, 5: 5}

@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_sha_required(terminate, algorithm):
    assert analyse(algorithm) == REQUIRED[algorithm]
    assert sha.information()['status'] == 0

@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_sha_recover(terminate, algorithm):
    # with one more pair than required, as in test_sha.py
    m = analyse(algorithm) + 1
    strans, ytrans = differences(H, m)
    val = recover(m, strans, ytrans)
    assert sha.information()['status'] == 0
    assert val == pytest.approx(val_h, abs=1e-8)

@pytest.mark.parametrize('order', ['reversed', None], ids=['reversed',
                                                            'None'])
@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_sha_order(terminate, algorithm, order):
    # the pairs may be used in any order, and order may be None
    m = analyse(algorithm) + 1
    strans, ytrans = differences(H, m)
    order = np.arange(m)[::-1].copy() if order == 'reversed' else None
    val = recover(m, strans, ytrans, order=order)
    assert sha.information()['status'] == 0
    assert val == pytest.approx(val_h, abs=1e-8)

@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_sha_larger_arrays(terminate, algorithm):
    # strans and ytrans may have more rows than the pairs used
    m = analyse(algorithm) + 1
    strans, ytrans = differences(H, m, rows=m + 3)
    val = recover(m, strans, ytrans)
    assert sha.information()['status'] == 0
    assert val == pytest.approx(val_h, abs=1e-8)

@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_sha_tridiagonal(terminate, algorithm):
    m = analyse(algorithm, row_t, col_t) + 1
    strans, ytrans = differences(H_t, m)
    val = recover(m, strans, ytrans, ne=len(row_t))
    assert sha.information()['status'] == 0
    assert val == pytest.approx(val_t, abs=1e-8)

def test_sha_insufficient(terminate):
    # with fewer pairs than required, the estimate may be inaccurate
    # (status 1)
    m = analyse(1) - 2
    strans, ytrans = differences(H, m)
    recover(m, strans, ytrans)
    assert sha.information()['status'] == 1
