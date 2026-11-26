from galahad import uls
import numpy as np
import pytest


# uls solves A x = b or A^T x = b for an m by n matrix A, by factorizing A.
# solve_system works in place on an array of length max(m,n), which holds
# b in its first m (n for A^T x = b) entries on entry, and x in its first
# n (m) entries on exit, and returns it. The LAPACK solver getr (LU with
# partial pivoting) can factorize rectangular matrices, but only solve
# square systems.

# 1) as in test_uls.py:
#      ( 1     )        ( 1 )                  ( 1 )
#  A = (   2 2 ),   b = ( 4 ) and A^T x = b' = ( 3 ) for x = ( 1, 1, 1 ).
#      (   1 3 )        ( 4 )                  ( 5 )
# 2) the same A, with x = ( 1, 2, 3 ): A x = ( 1, 10, 11 ) and
# A^T x = ( 1, 7, 13 )
A = np.array([[1.0, 0.0, 0.0],
              [0.0, 2.0, 2.0],
              [0.0, 1.0, 3.0]])
m = n = 3
b_e = np.array([1.0, 4.0, 4.0])
b_et = np.array([1.0, 3.0, 5.0])
x_star_e = np.array([1.0, 1.0, 1.0])
x_star = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 10.0, 11.0])
b_t = np.array([1.0, 7.0, 13.0])

# 3) A singular of rank 2
A_s = np.array([[1.0, 2.0, 0.0],
                [2.0, 4.0, 0.0],
                [0.0, 0.0, 1.0]])

# 4) the 3 x 2 matrix A_r = ( 1 0 ; 0 2 ; 1 1 ) and its transpose, both of
# rank 2
A_r = np.array([[1.0, 0.0],
                [0.0, 2.0],
                [1.0, 1.0]])

STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

def storage(A, A_type):
    # return A_ne, A_row, A_col, A_ptr, A_val for A
    if A_type == 'dense':
        return A.size, None, None, None, A.ravel().copy()
    rows, cols = np.nonzero(A)
    if A_type == 'coordinate':
        return len(rows), rows, cols, None, A[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=A.shape[0]))]
    return len(rows), None, cols, ptr, A[rows, cols]

def options():
    o = uls.initialize('getr')
    o['print_level'] = 0
    return o

@pytest.fixture
def terminate():
    yield
    uls.terminate()

def factorize(A, A_type='coordinate'):
    m, n = A.shape
    A_ne, A_row, A_col, A_ptr, A_val = storage(A, A_type)
    uls.factorize_matrix(m, n, A_type, A_ne, A_row, A_col, A_ptr, A_val,
                         options())

def solve(b, trans=False):
    # solve A x = b (A^T x = b) for the square A factorized
    bc = b.copy()
    x = uls.solve_system(len(b), len(b), bc, trans)
    return x, bc


@pytest.fixture(params=[False, True], ids=['A', 'A^T'])
def solved(request, terminate):
    # problem 2)
    factorize(A)
    trans = request.param
    x, bc = solve(b_t if trans else b, trans)
    return x, bc, uls.information(), trans

def test_uls_status(solved):
    x, bc, inform, trans = solved
    assert inform['status'] == 0
    assert inform['solver'].strip() == 'getr'

def test_uls_solution(solved):
    x, bc, inform, trans = solved
    assert x == pytest.approx(x_star, abs=1e-8)
    if trans:
        assert A.T @ x == pytest.approx(b_t, abs=1e-8)
    else:
        assert A @ x == pytest.approx(b, abs=1e-8)

def test_uls_in_place(solved):
    # solve_system overwrites b by x, and returns b
    x, bc, inform, trans = solved
    assert x is bc

def test_uls_inform(solved):
    x, bc, inform, trans = solved
    assert inform['rank'] == 3


@pytest.mark.parametrize('trans', [False, True], ids=['A', 'A^T'])
def test_uls_example(terminate, trans):
    # problem 1)
    factorize(A)
    x, bc = solve(b_et if trans else b_e, trans)
    assert uls.information()['status'] == 0
    assert x == pytest.approx(x_star_e, abs=1e-8)

@pytest.mark.parametrize('trans', [False, True], ids=['A', 'A^T'])
@pytest.mark.parametrize('A_type', STORAGE)
def test_uls_storage(terminate, A_type, trans):
    factorize(A, A_type)
    x, bc = solve(b_t if trans else b, trans)
    assert uls.information()['status'] == 0
    assert x == pytest.approx(x_star, abs=1e-8)

def test_uls_refactorize(terminate):
    # problem 2) with A doubled, after a factorization of A
    factorize(A)
    factorize(2.0 * A)
    x, bc = solve(b)
    assert uls.information()['status'] == 0
    assert x == pytest.approx(x_star / 2.0, abs=1e-8)

def test_uls_solve_repeated(terminate):
    # several solves with one factorization
    factorize(A)
    for rhs, x_ref, trans in [(b, x_star, False), (b_e, x_star_e, False),
                              (b_t, x_star, True), (b_et, x_star_e, True)]:
        x, bc = solve(rhs, trans)
        assert uls.information()['status'] == 0
        assert x == pytest.approx(x_ref, abs=1e-8)


@pytest.mark.parametrize('A_type', STORAGE)
def test_uls_singular_rank(terminate, A_type):
    # problem 3): getr factorizes a singular A, and reports its rank. What a
    # solve should then give is not settled (see
    # https://github.com/ralna/GALAHAD/issues/655), so it is not tested
    factorize(A_s, A_type)
    inform = uls.information()
    assert inform['status'] == 0
    assert inform['rank'] == 2


RECT_PR = 'https://github.com/ralna/GALAHAD/pull/654'
RECTANGULAR = [(A_r, 'coordinate'), (A_r, 'sparse_by_rows'), (A_r, 'dense'),
               (A_r.T, 'coordinate'), (A_r.T, 'sparse_by_rows'),
               (A_r.T, 'dense')]
RECTANGULAR_IDS = [f'{mm}x{nn}-{t}' for (M, t) in RECTANGULAR
                   for mm, nn in [M.shape]]

@pytest.mark.parametrize('A_r, A_type', RECTANGULAR, ids=RECTANGULAR_IDS)
def test_uls_rectangular_rank(terminate, A_r, A_type):
    # problem 4): getr factorizes rectangular matrices, and reports their rank
    factorize(A_r, A_type)
    inform = uls.information()
    assert inform['status'] == 0
    assert inform['rank'] == 2

# getr writes to unallocated workspace when solving with a rectangular A,
# which crashes Python, so this cannot be a strict xfail
@pytest.mark.skip(reason='uls crashes when getr solves with a rectangular '
                  f'matrix (see {RECT_PR})')
@pytest.mark.parametrize('trans', [False, True], ids=['A', 'A^T'])
@pytest.mark.parametrize('A_r', [A_r, A_r.T], ids=['3x2', '2x3'])
def test_uls_rectangular_solve(terminate, A_r, trans):
    # getr cannot solve rectangular systems (status -29)
    factorize(A_r)
    with pytest.warns(RuntimeWarning):
        uls.solve_system(*A_r.shape, np.ones(3), trans)
    assert uls.information()['status'] == -29

@pytest.mark.xfail(strict=True, reason='uls_factorize_matrix overwrites the '
                   f'status from the factorization (see {RECT_PR})')
def test_uls_no_rows(terminate):
    # A must have at least one row (status -3)
    with pytest.raises(ValueError):
        uls.factorize_matrix(0, n, 'dense', 0, None, None, None, np.zeros(0),
                             options())
    assert uls.information()['status'] == -3
