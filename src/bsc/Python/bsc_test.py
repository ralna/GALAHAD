from galahad import bsc
import numpy as np
import pytest


# bsc forms the lower triangle of S = A D A^T, for an m by n matrix A and
# a diagonal D (I if omitted), returning it by co-ordinates and by rows

# 1) as in test_bsc.py: A = ( 1 1 0 0 ; 0 0 1 1 ; 1 0 0 1 ) and
# D = diag( 1, 2, 3, 4 ), so S = ( 3 0 1 ; 0 7 4 ; 1 4 5 ), whose lower
# triangle has 5 entries
A = np.array([[1.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 1.0],
              [1.0, 0.0, 0.0, 1.0]])
D = np.array([1.0, 2.0, 3.0, 4.0])
S_ref = np.array([[3.0, 0.0, 1.0],
                  [0.0, 7.0, 4.0],
                  [1.0, 4.0, 5.0]])

STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

def storage(A, A_type):
    # return A_ne, A_row, A_col, A_ptr, A_val for A
    m = A.shape[0]
    if A_type == 'dense':
        return A.size, None, None, None, A.ravel().copy()
    rows, cols = np.nonzero(A)
    if A_type == 'coordinate':
        return len(rows), rows, cols, None, A[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=m))]
    return len(rows), None, cols, ptr, A[rows, cols]

@pytest.fixture
def terminate():
    yield
    bsc.terminate()

def form(A, D, A_type='coordinate'):
    # return S_ne, S_row, S_col, S_ptr, S_val
    m, n = A.shape
    A_ne, A_row, A_col, A_ptr, A_val = storage(A, A_type)
    o = bsc.initialize()
    o['print_level'] = 0
    S_ne = bsc.load(m, n, A_type, A_ne, A_row, A_col, A_ptr, o)
    return (S_ne,) + bsc.form(m, n, A_ne, A_val, S_ne, D)

def check_S(m, S_ne, S_row, S_col, S_ptr, S_val, S_ref):
    # S_row, S_col and S_val give the lower triangle of S
    assert len(S_row) == S_ne
    assert np.all(S_row >= S_col)
    S = np.zeros((m, m))
    S[S_row, S_col] = S_val
    assert S == pytest.approx(np.tril(S_ref), abs=1e-12)
    assert S_ptr[m] == S_ne


@pytest.fixture
def formed(terminate):
    # problem 1)
    return form(A, D)

def test_bsc_status(formed):
    assert bsc.information()['status'] == 0

def test_bsc_number_of_entries(formed):
    S_ne = formed[0]
    assert S_ne == 5

def test_bsc_matrix(formed):
    check_S(3, *formed, S_ref)

@pytest.mark.parametrize('A_type', STORAGE)
def test_bsc_storage(terminate, A_type):
    check_S(3, *form(A, D, A_type), S_ref)

# S_ptr should give the start of each row of the lower triangle of S (with
# S_col and S_val in row-wise order), but for sparse A the entries are
# stored by columns
BSC_ISSUE = 'https://github.com/ralna/GALAHAD/issues/661'

@pytest.mark.parametrize('A_type', [
    pytest.param(t, marks=pytest.mark.xfail(
        strict=True, reason='bsc stores S by columns for sparse A '
        f'(see {BSC_ISSUE})')) for t in STORAGE[:2]] + ['dense'])
def test_bsc_by_rows(terminate, A_type):
    S_ne, S_row, S_col, S_ptr, S_val = form(A, D, A_type)
    for i in range(3):
        assert np.all(S_row[S_ptr[i]:S_ptr[i + 1]] == i)

def test_bsc_identity(terminate):
    # D = None means D = I
    check_S(3, *form(A, None), A @ A.T)

def test_bsc_keywords(terminate):
    # m and n may be given by keyword
    A_ne, A_row, A_col, A_ptr, A_val = storage(A, 'coordinate')
    o = bsc.initialize()
    o['print_level'] = 0
    S_ne = bsc.load(m=3, n=4, A_type='coordinate', A_ne=A_ne, A_row=A_row,
                    A_col=A_col, A_ptr=A_ptr, options=o)
    S = bsc.form(m=3, n=4, A_ne=A_ne, A_val=A_val, S_ne=S_ne, D=D)
    check_S(3, S_ne, *S, S_ref)

@pytest.mark.parametrize('A_type', STORAGE)
def test_bsc_more_rows_than_columns(terminate, A_type):
    # A^T is 4 by 3, so S = A^T D_3 A is 4 by 4
    D_3 = np.array([1.0, 2.0, 3.0])
    check_S(4, *form(A.T.copy(), D_3, A_type), A.T @ np.diag(D_3) @ A)

def test_bsc_refactor(terminate):
    # a second formation with new values for A and D
    m, n = A.shape
    A_ne, A_row, A_col, A_ptr, A_val = storage(A, 'coordinate')
    o = bsc.initialize()
    o['print_level'] = 0
    S_ne = bsc.load(m, n, 'coordinate', A_ne, A_row, A_col, A_ptr, o)
    bsc.form(m, n, A_ne, A_val, S_ne, D)
    S = bsc.form(m, n, A_ne, 2.0 * A_val, S_ne, 2.0 * D)
    assert bsc.information()['status'] == 0
    check_S(m, S_ne, *S, 8.0 * S_ref)
