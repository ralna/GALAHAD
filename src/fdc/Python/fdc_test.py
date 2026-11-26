from galahad import fdc
import numpy as np
import pytest


# fdc finds rows of A that depend on the others, so that removing them
# leaves a matrix of full row rank, and checks that A x = b is consistent.
# It returns their number n_depen and their indices (starting from 0) in the
# first n_depen entries of depen

# 1) as in test_fdc.py: the second row of A is 2 times the first minus 1.6
# times the third, so A has rank 2, and b = ( 5, 10, 0 ) satisfies the same
# relation, so A x = b is consistent; any one row may be removed
A = np.array([[1.0, 2.0, 3.0, 4.0],
              [2.0, -4.0, 6.0, -8.0],
              [0.0, 5.0, 0.0, 10.0]])
b = np.array([5.0, 10.0, 0.0])

def options():
    o = fdc.initialize()
    o['print_level'] = 0
    # use the LAPACK solver sytr rather than the default, which needs the
    # environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    o['use_sls'] = True
    o['symmetric_linear_solver'] = 'sytr'
    return o

@pytest.fixture
def terminate():
    yield
    fdc.terminate()

def find(A, b):
    # A is given by rows
    m, n = A.shape
    rows, cols = np.nonzero(A)
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=m))]
    return fdc.find_dependent_rows(m, n, ptr, cols, A[rows, cols].copy(), b,
                                   options())

def check_independent(A, n_depen, depen):
    # the remaining rows have full rank, equal to that of A
    keep = np.setdiff1d(np.arange(A.shape[0]), depen[:n_depen])
    assert np.linalg.matrix_rank(A[keep]) == len(keep)
    assert len(keep) == np.linalg.matrix_rank(A)


@pytest.fixture
def found(terminate):
    # problem 1)
    return find(A, b)

def test_fdc_status(found):
    n_depen, depen, inform = found
    assert inform['status'] == 0

def test_fdc_dependent(found):
    n_depen, depen, inform = found
    assert n_depen == 1
    assert len(depen) == A.shape[0]
    check_independent(A, n_depen, depen)

def test_fdc_inconsistent(terminate):
    # b = ( 5, 11, 0 ) does not satisfy the relation (status -5, which is a
    # warning)
    with pytest.warns(RuntimeWarning):
        n_depen, depen, inform = find(A, np.array([5.0, 11.0, 0.0]))
    assert inform['status'] == -5

def test_fdc_full_rank(terminate):
    A_f = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0]])
    n_depen, depen, inform = find(A_f, np.array([1.0, 2.0, 3.0]))
    assert inform['status'] == 0
    assert n_depen == 0

def test_fdc_zero_row(terminate):
    # a row without entries (and a zero right-hand side) is dependent
    A_z = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]])
    n_depen, depen, inform = find(A_z, np.array([1.0, 0.0, 2.0]))
    assert inform['status'] == 0
    assert n_depen == 1
    assert depen[0] == 1

def test_fdc_more_rows_than_columns(terminate):
    # a 5 x 2 matrix of rank 2 has 3 dependent rows, more than its number of
    # columns
    A_t = np.array([[1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                    [1.0, -1.0]])
    n_depen, depen, inform = find(A_t, A_t @ np.array([1.0, 2.0]))
    assert inform['status'] == 0
    assert n_depen == 3
    check_independent(A_t, n_depen, depen)
