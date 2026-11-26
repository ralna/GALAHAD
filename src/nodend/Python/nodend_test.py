from galahad import nodend
import numpy as np
import pytest


# nodend computes a nested-dissection ordering of a sparse symmetric matrix
# A (given by its lower triangle), to reduce the fill-in when factorizing
# P A P^T. perm[i] is the position, in P A P^T, of row (and column) i of A

# 1) as in test_nodend.py: a 3 by 3 matrix
A_e = np.array([[1.0, 0.0, 0.0],
                [0.0, 2.0, 1.0],
                [0.0, 1.0, 3.0]])

# 2) an "arrow" matrix, whose first row and column are full. Eliminating
# that row first fills in the whole Cholesky factor, but eliminating it
# last gives no fill-in
N = 6
A_a = 4.0 * np.eye(N)
A_a[1:, 0] = 1.0
A_a[0, 1:] = 1.0

# 3) the 2D Laplacian on a 4 by 4 grid
k = 4
A_l = 4.0 * np.eye(k * k)
for i in range(k):
    for j in range(k):
        p = i * k + j
        if j < k - 1:
            A_l[p, p + 1] = A_l[p + 1, p] = -1.0
        if i < k - 1:
            A_l[p, p + k] = A_l[p + k, p] = -1.0

STORAGE = ['coordinate', 'sparse_by_rows']

@pytest.fixture
def terminate():
    yield
    nodend.terminate()

def order(A, A_type='coordinate'):
    n = A.shape[0]
    o = nodend.initialize()
    o['print_level'] = 0
    if A_type == 'dense':
        return nodend.order(n, A_type, n * (n + 1) // 2, None, None, None, o)
    rows, cols = np.nonzero(np.tril(A))
    if A_type == 'coordinate':
        return nodend.order(n, A_type, len(rows), rows, cols, None, o)
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))]
    return nodend.order(n, A_type, len(rows), None, cols, ptr, o)

def is_permutation(perm, n):
    return sorted(perm) == list(range(n))

def fill(A, perm):
    # the number of nonzeros in the Cholesky factor of P A P^T, where row i
    # of A is row perm[i] of P A P^T
    sequence = np.argsort(perm)
    L = np.linalg.cholesky(A[np.ix_(sequence, sequence)])
    return int(np.count_nonzero(np.abs(L) > 1e-12))


def test_nodend_example(terminate):
    perm = order(A_e)
    assert nodend.information()['status'] == 0
    assert is_permutation(perm, 3)

@pytest.mark.parametrize('A_type', STORAGE)
def test_nodend_arrow(terminate, A_type):
    perm = order(A_a, A_type)
    assert nodend.information()['status'] == 0
    assert is_permutation(perm, N)
    # the full row and column are moved last, so that there is no fill-in
    assert perm[0] == N - 1
    assert fill(A_a, perm) == np.count_nonzero(np.tril(A_a))
    assert fill(A_a, np.arange(N)) == N * (N + 1) // 2

@pytest.mark.parametrize('A_type', STORAGE)
def test_nodend_laplacian(terminate, A_type):
    perm = order(A_l, A_type)
    assert nodend.information()['status'] == 0
    assert is_permutation(perm, k * k)
    # nested dissection gives less fill-in than the natural ordering
    assert fill(A_l, perm) < fill(A_l, np.arange(k * k))

def test_nodend_storage_agree(terminate):
    # the co-ordinate and sparse-by-rows schemes give the same ordering
    perm_c = order(A_l, 'coordinate')
    nodend.terminate()
    perm_r = order(A_l, 'sparse_by_rows')
    assert np.array_equal(perm_c, perm_r)

def test_nodend_dense(terminate):
    # there is nothing to gain for a dense matrix, which keeps its order
    perm = order(A_l, 'dense')
    assert nodend.information()['status'] == 0
    assert np.array_equal(perm, np.arange(k * k))
