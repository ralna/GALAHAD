from galahad import sls
import numpy as np
import pytest


# sls solves A x = b for a symmetric matrix A, given its lower triangle,
# by factorizing A = P L D L^T P^T. solve_system overwrites b by x, and
# partial_solve_system solves one of P L x = b ('L'), D x = b ('D'),
# L^T P^T x = b ('U') or P L sqrt( D ) x = b ('S'), again in place. Both
# return b. The LAPACK solvers are sytr (L D L^T with 1x1 and 2x2 pivots)
# and potr (Cholesky, A = L L^T, so 'D' is the identity and 'S' is 'L').

# 1) as in test_sls.py:
#      ( 1     )        ( 1 )
#  A = (   2 1 ),   b = ( 3 ),   x = ( 1, 1, 1 ).
#      (   1 3 )        ( 4 )
# Without pivoting, A = L D L^T with L = I + e_3 e_2^T / 2 and
# D = diag( 1, 2, 5/2 ), so L^{-1} b = ( 1, 3, 5/2 ),
# D^{-1} L^{-1} b = ( 1, 3/2, 1 ) and L^{-T} of that is x. The Cholesky
# factor is L sqrt( D ), and ( L sqrt( D ) )^{-1} b = ( 1, 3/sqrt(2),
# sqrt(5/2) ), whose squared norm is b^T A^{-1} b = b^T x = 8
A = np.array([[1.0, 0.0, 0.0],
              [0.0, 2.0, 1.0],
              [0.0, 1.0, 3.0]])
b = np.array([1.0, 3.0, 4.0])
x_star = np.array([1.0, 1.0, 1.0])
n = 3
b_l = np.array([1.0, 3.0, 2.5])
b_dl = np.array([1.0, 1.5, 1.0])
b_s = np.array([1.0, 3.0 / np.sqrt(2.0), np.sqrt(2.5)])

# 2) A indefinite: a_22 = -2. Then x = ( 1, -5/7, 11/7 ), and A has one
# negative eigenvalue
A_i = np.array([[1.0, 0.0, 0.0],
                [0.0, -2.0, 1.0],
                [0.0, 1.0, 3.0]])
x_star_i = np.array([1.0, -5.0 / 7.0, 11.0 / 7.0])

# 3) A singular of rank 2: a_22 = a_23 = a_33 = 1. A x = b has solutions
# x = ( 1, t, 3 - t ) for b = ( 1, 3, 3 ), and none for b = ( 1, 3, 4 )
A_s = np.array([[1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0]])
b_c = np.array([1.0, 3.0, 3.0])

# 4) A = ( 0 1 ; 1 0 ), which sytr factorizes with a single 2x2 pivot, so
# L = I and D = A, and A has one negative eigenvalue. sqrt( D ) does not
# exist
A_2 = np.array([[0.0, 1.0],
                [1.0, 0.0]])
b_2 = np.array([1.0, 2.0])
x_star_2 = np.array([2.0, 1.0])

SOLVERS = ['sytr', 'potr']
STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

def storage(A, A_type):
    # return A_ne, A_row, A_col, A_ptr, A_val for the lower triangle of A
    n = A.shape[0]
    if A_type == 'dense':
        rows, cols = np.tril_indices(n)
        return len(rows), None, None, None, A[rows, cols]
    rows, cols = np.nonzero(np.tril(A))
    if A_type == 'coordinate':
        return len(rows), rows, cols, None, A[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))]
    return len(rows), None, cols, ptr, A[rows, cols]

def options(solver):
    o = sls.initialize(solver)
    o['print_level'] = 0
    return o

@pytest.fixture
def terminate():
    yield
    sls.terminate()

def factorize(A, solver='sytr', A_type='coordinate', A_data=None):
    # analyse and factorize A. A_data = ( type, ne, val ) gives A directly,
    # for the storage schemes without indices
    if A_data is None:
        n = A.shape[0]
        A_ne, A_row, A_col, A_ptr, A_val = storage(A, A_type)
    else:
        n = 3
        A_type, A_ne, A_val = A_data
        A_row = A_col = A_ptr = None
    sls.analyse_matrix(n, A_type, A_ne, A_row, A_col, A_ptr, options(solver))
    sls.factorize_matrix(A_ne, A_val)
    return n


@pytest.fixture(params=SOLVERS)
def solved(request, terminate):
    # problem 1)
    factorize(A, request.param)
    bc = b.copy()
    x = sls.solve_system(n, bc)
    return x, bc, sls.information(), request.param

def test_sls_status(solved):
    x, bc, inform, solver = solved
    assert inform['status'] == 0
    assert inform['solver'].strip() == solver

def test_sls_solution(solved):
    x, bc, inform, solver = solved
    assert x == pytest.approx(x_star, abs=1e-8)
    assert A @ x == pytest.approx(b, abs=1e-8)

def test_sls_in_place(solved):
    # solve_system overwrites b by x, and returns b
    x, bc, inform, solver = solved
    assert x is bc

def test_sls_inform(solved):
    x, bc, inform, solver = solved
    assert inform['rank'] == 3
    assert inform['negative_eigenvalues'] == 0
    assert inform['two_by_two_pivots'] == 0
    assert inform['entries_in_factors'] == 6


@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('A_type', STORAGE)
def test_sls_storage(terminate, A_type, solver):
    factorize(A, solver, A_type)
    x = sls.solve_system(n, b.copy())
    assert sls.information()['status'] == 0
    assert x == pytest.approx(x_star, abs=1e-8)


# 5) the storage schemes without indices: A = diag( 1, 2, 4 ) gives
# x = ( 1, 3/2, 1 ), A = 2 I gives b / 2 and A = I gives b. For A = 0
# there is no solution (status -11)
A_TRIVIAL = [(('diagonal', 3, np.array([1.0, 2.0, 4.0])),
              np.array([1.0, 1.5, 1.0])),
             (('scaled_identity', 1, np.array([2.0])), b / 2.0),
             (('identity', 0, np.zeros(0)), b),
             (('identity', 0, None), b)]

@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('A_data, x_ref', A_TRIVIAL,
                         ids=['diagonal', 'scaled_identity', 'identity',
                              'identity-None'])
def test_sls_trivial(terminate, A_data, x_ref, solver):
    factorize(None, solver, A_data=A_data)
    x = sls.solve_system(n, b.copy())
    inform = sls.information()
    assert inform['status'] == 0
    assert inform['rank'] == 3
    assert x == pytest.approx(x_ref, abs=1e-8)

@pytest.mark.parametrize('A_val', [np.zeros(0), None], ids=['', 'None'])
@pytest.mark.parametrize('A_type', ['zero', 'none'])
def test_sls_zero(terminate, A_type, A_val):
    factorize(None, A_data=(A_type, 0, A_val))
    with pytest.warns(RuntimeWarning):
        sls.solve_system(n, b.copy())
    inform = sls.information()
    assert inform['status'] == -11
    assert inform['rank'] == 0


def test_sls_indefinite(terminate):
    # problem 2) with sytr
    factorize(A_i, 'sytr')
    x = sls.solve_system(n, b.copy())
    inform = sls.information()
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_i, abs=1e-8)
    assert inform['rank'] == 3
    assert inform['negative_eigenvalues'] == 1

def test_sls_indefinite_potr(terminate):
    # problem 2): potr needs A positive definite (status -20)
    with pytest.warns(RuntimeWarning):
        factorize(A_i, 'potr')
    assert sls.information()['status'] == -20


def test_sls_singular_consistent(terminate):
    # problem 3) with a consistent b, using sytr
    factorize(A_s, 'sytr')
    x = sls.solve_system(n, b_c.copy())
    inform = sls.information()
    assert inform['status'] == 0
    assert inform['rank'] == 2
    assert A_s @ x == pytest.approx(b_c, abs=1e-8)

def test_sls_singular_inconsistent(terminate):
    # problem 3) with an inconsistent b, using sytr (status -5)
    factorize(A_s, 'sytr')
    with pytest.warns(RuntimeWarning):
        sls.solve_system(n, b.copy())
    inform = sls.information()
    assert inform['status'] == -5
    assert inform['rank'] == 2


# partial solves of problem 1): 'L', 'D', 'U' and 'S' give the vectors
# above; for potr, 'D' leaves b unchanged and 'S' is 'L'
PARTS = {'sytr': {'L': b_l, 'D': b / np.array([1.0, 2.0, 2.5]), 'S': b_s},
         'potr': {'L': b_s, 'D': b, 'S': b_s}}

@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('part', ['L', 'D', 'S'])
def test_sls_partial_solve(terminate, part, solver):
    factorize(A, solver)
    bc = b.copy()
    x = sls.partial_solve_system(part, n, bc)
    assert sls.information()['status'] == 0
    assert x is bc
    assert x == pytest.approx(PARTS[solver][part], abs=1e-8)

@pytest.mark.parametrize('solver', SOLVERS)
def test_sls_partial_solve_sequence(terminate, solver):
    # 'L', then 'D', then 'U' give x
    factorize(A, solver)
    bc = b.copy()
    for part in 'LDU':
        sls.partial_solve_system(part, n, bc)
        assert sls.information()['status'] == 0
    assert bc == pytest.approx(x_star, abs=1e-8)

@pytest.mark.parametrize('solver', SOLVERS)
def test_sls_partial_solve_u(terminate, solver):
    # 'U' applied to D^{-1} L^{-1} b (sytr) or L^{-1} b (potr) gives x
    factorize(A, solver)
    bc = (b_dl if solver == 'sytr' else b_s).copy()
    sls.partial_solve_system('U', n, bc)
    assert sls.information()['status'] == 0
    assert bc == pytest.approx(x_star, abs=1e-8)

def test_sls_partial_solve_2x2(terminate):
    # problem 4) with sytr: L = I and D = A
    factorize(A_2, 'sytr')
    inform = sls.information()
    assert inform['two_by_two_pivots'] == 1
    assert inform['negative_eigenvalues'] == 1
    bc = b_2.copy()
    sls.partial_solve_system('L', 2, bc)
    assert bc == pytest.approx(b_2, abs=1e-8)
    sls.partial_solve_system('D', 2, bc)
    assert bc == pytest.approx(x_star_2, abs=1e-8)
    sls.partial_solve_system('U', 2, bc)
    assert sls.information()['status'] == 0
    assert bc == pytest.approx(x_star_2, abs=1e-8)

@pytest.mark.parametrize('problem', [(A_2, b_2), (A_i, b)],
                         ids=['2x2', 'negative'])
def test_sls_partial_solve_s_indefinite(terminate, problem):
    # 'S' needs D positive definite (status -20)
    A, b = problem
    factorize(A, 'sytr')
    with pytest.warns(RuntimeWarning):
        sls.partial_solve_system('S', A.shape[0], b.copy())
    assert sls.information()['status'] == -20


# sls treats an unrecognised part as 'S' (sytr) or 'D' (potr), so only
# the other parts show that lower case is ignored
PART_PR = 'https://github.com/ralna/GALAHAD/pull/653'

@pytest.mark.xfail(strict=True, reason='sls ignores lower-case parts '
                   f'(see {PART_PR})')
@pytest.mark.parametrize('solver, part', [('sytr', 'L'), ('sytr', 'D'),
                                          ('potr', 'L'), ('potr', 'S')])
def test_sls_partial_solve_lower_case(terminate, part, solver):
    factorize(A, solver)
    x = sls.partial_solve_system(part.lower(), n, b.copy())
    assert sls.information()['status'] == 0
    assert x == pytest.approx(PARTS[solver][part], abs=1e-8)

@pytest.mark.xfail(strict=True, reason='sls does not reject an unknown part '
                   f'(see {PART_PR})')
@pytest.mark.parametrize('solver', SOLVERS)
def test_sls_partial_solve_unknown(terminate, solver):
    # an unknown part is rejected (status -3)
    factorize(A, solver)
    with pytest.raises(ValueError):
        sls.partial_solve_system('X', n, b.copy())
    assert sls.information()['status'] == -3


def test_sls_refactorize(terminate):
    # problem 2) after problem 1), with the same sparsity pattern
    factorize(A, 'sytr')
    A_ne, A_row, A_col, A_ptr, A_val = storage(A_i, 'coordinate')
    sls.factorize_matrix(A_ne, A_val)
    x = sls.solve_system(n, b.copy())
    inform = sls.information()
    assert inform['status'] == 0
    assert inform['negative_eigenvalues'] == 1
    assert x == pytest.approx(x_star_i, abs=1e-8)
