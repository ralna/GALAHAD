from galahad import llst
import numpy as np
import pytest


# llst finds a global minimizer of the linear least-squares objective
# 1/2 ||A x - b||^2 within the trust region ||x||_S <= radius, where
# ||x||_S = sqrt( x^T S x ), using factorizations. x solves the problem if,
# for some lambda >= 0, ( A^T A + lambda S ) x = A^T b,
# lambda ( ||x||_S - radius ) = 0 and ||x||_S <= radius. inform reports
# lambda ('multiplier'), ||x||_S ('x_norm') and ||A x - b|| ('r_norm').

# 1) as in test_llst.py: A = ( 1 1 1 ), b = 1, radius 1/10 and S = I. The
# minimum-norm solution of A x = b, ( 1, 1, 1 ) / 3, lies outside the
# region, so x = alpha ( 1, 1, 1 ) with ||x|| = sqrt( 3 ) alpha = 1/10 and
# ( 3 + lambda ) alpha = 1, i.e., lambda = 10 sqrt( 3 ) - 3. With
# S = diag( 1, 2, 1 ) as there, lambda has no closed form, so check the
# conditions above
A_e = np.ones((1, 3))
b_e = np.array([1.0])
S_e = np.diag([1.0, 2.0, 1.0])
radius_e = 0.1
alpha_e = radius_e / np.sqrt(3.0)
lambda_e = 1.0 / alpha_e - 3.0

# 2) a small problem built from a chosen x and lambda:
# A = ( I ; diag( 1, 2 ) ), so that A^T A = diag( 2, 5 ):
# - S = I and b = ( 9/10, 4/5, 9/10, 2 ), so that A^T b = ( 9/5, 24/5 ):
#   with radius 1, x = ( 3, 4 ) / 5 has ||x|| = 1 and ( A^T A + I ) x = A^T b,
#   so lambda = 1, with ||A x - b|| = sqrt( 34 ) / 10; with radius 3/2, the
#   unconstrained minimizer ( A^T A )^-1 A^T b = ( 9/10, 24/25 ), of norm
#   1.316, lies inside the region, so lambda = 0, with
#   ||A x - b|| = 2 sqrt( 5 ) / 25;
# - S = diag( 1, 4 ), b = ( 9/10, 6/5, 9/10, 6/5 ), so that
#   A^T b = ( 9/5, 18/5 ), and radius 1: x = ( 3, 2 ) / 5 has ||x||_S = 1,
#   lambda = 1 and ( A^T A + S ) x = A^T b, with ||A x - b||^2 = 98/100;
# - S = 4 I, the same b and radius 1: x = ( 3, 4 ) / 10 has ||x||_S = 1,
#   lambda = 1 and ( A^T A + 4 I ) x = A^T b, with ||A x - b||^2 = 152/100
A = np.vstack([np.eye(2), np.diag([1.0, 2.0])])
b = np.array([0.9, 0.8, 0.9, 2.0])
PROBLEMS = {
    'boundary': dict(radius=1.0, x=np.array([0.6, 0.8]), lam=1.0,
                     r_norm=np.sqrt(34.0) / 10.0),
    'interior': dict(radius=1.5, x=np.array([0.9, 0.96]), lam=0.0,
                     r_norm=2.0 * np.sqrt(5.0) / 25.0),
}
b_s = np.array([0.9, 1.2, 0.9, 1.2])
S = np.diag([1.0, 4.0])
x_star_s = np.array([0.6, 0.4])
x_star_4 = np.array([0.3, 0.4])

A_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

# with the default (Schur-complement) SBLS factorization, llst misses
# interior solutions (and, for larger radii, does not terminate)
INTERIOR_XFAIL = pytest.mark.xfail(
    strict=True, reason='llst misses interior solutions with the default '
    'SBLS factorization (see https://github.com/ralna/GALAHAD/issues/643)')
# and gives wrong boundary solutions for S in coordinate or sparse-by-rows
# storage
SPARSE_S_XFAIL = pytest.mark.xfail(
    strict=True, reason='llst gives wrong solutions for sparse S with the '
    'default SBLS factorization (see '
    'https://github.com/ralna/GALAHAD/issues/643)')

def a_storage(A, A_type):
    # return A_ne, A_row, A_col, A_ptr, A_val for A
    if A_type == 'dense':
        return A.size, None, None, None, A.ravel()
    rows, cols = np.nonzero(A)
    if A_type == 'coordinate':
        return len(rows), rows, cols, None, A[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=A.shape[0]))]
    return len(rows), None, cols, ptr, A[rows, cols]

def s_storage(S, S_type):
    # return S_ne, S_row, S_col, S_ptr, S_val for the lower triangle of S
    if S_type == 'dense':
        rows, cols = np.tril_indices(S.shape[0])
        return len(rows), None, None, None, S[rows, cols]
    if S_type == 'diagonal':
        return S.shape[0], None, None, None, np.diag(S).copy()
    return a_storage(np.tril(S), S_type)

def options(**sbls):
    o = llst.initialize()
    o['print_level'] = 0
    # use the LAPACK solvers sytr and potr rather than the defaults, which
    # need the environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    o['definite_linear_solver'] = 'potr'
    o['sbls_options']['symmetric_linear_solver'] = 'sytr'
    o['sbls_options']['definite_linear_solver'] = 'potr'
    o['sbls_options'].update(sbls)
    return o

def solve(A, b, radius, A_type='coordinate', S=None, S_type='coordinate',
          S_data=None, **sbls):
    # solve the problem, returning x and inform. S_data = ( type, ne, val )
    # gives S directly, for the storage schemes without indices
    m, n = A.shape
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, A_type)
    llst.load(m, n, A_type, A_ne, A_row, A_col, A_ptr, options(**sbls))
    args = {}
    if S_data is not None:
        S_type, S_ne, S_val = S_data
        llst.load_scaling(n, S_type, S_ne, None, None, None)
        args = dict(S_ne=S_ne, S_val=S_val)
    elif S is not None:
        S_ne, S_row, S_col, S_ptr, S_val = s_storage(S, S_type)
        llst.load_scaling(n, S_type, S_ne, S_row, S_col, S_ptr)
        args = dict(S_ne=S_ne, S_val=S_val)
    x = llst.solve_problem(m, n, radius, A_ne, A_val, b, **args)
    return x, llst.information()

def check_solution(x, inform, radius, A, b, S=None):
    # check the conditions above for the reported multiplier, and the
    # reported norms
    S = np.eye(len(x)) if S is None else S
    lam = inform['multiplier']
    norm_x = np.sqrt(x @ S @ x)
    assert lam >= 0.0
    assert norm_x <= radius + 1e-8
    assert lam * (norm_x - radius) == pytest.approx(0.0, abs=1e-8)
    assert (A.T @ A + lam * S) @ x == pytest.approx(A.T @ b, abs=1e-8)
    assert inform['x_norm'] == pytest.approx(norm_x, abs=1e-8)
    assert inform['r_norm'] == pytest.approx(np.linalg.norm(A @ x - b),
                                             abs=1e-8)


@pytest.fixture
def terminate():
    yield
    llst.terminate()


def test_llst_example(terminate):
    # problem 1) with S = I
    x, inform = solve(A_e, b_e, radius_e)
    assert inform['status'] == 0
    assert x == pytest.approx(alpha_e * np.ones(3), abs=1e-8)
    assert inform['multiplier'] == pytest.approx(lambda_e, abs=1e-8)
    assert inform['r_norm'] == pytest.approx(1.0 - 3.0 * alpha_e, abs=1e-8)
    check_solution(x, inform, radius_e, A_e, b_e)


def test_llst_example_scaled(terminate):
    # problem 1) with S = diag( 1, 2, 1 )
    x, inform = solve(A_e, b_e, radius_e, S=S_e)
    assert inform['status'] == 0
    assert inform['x_norm'] == pytest.approx(radius_e, abs=1e-8)
    check_solution(x, inform, radius_e, A_e, b_e, S_e)


@pytest.mark.parametrize('A_type', A_STORAGE)
@pytest.mark.parametrize('problem', [
    'boundary', pytest.param('interior', marks=INTERIOR_XFAIL)])
def test_llst(terminate, problem, A_type):
    # problem 2) with S = I
    p = PROBLEMS[problem]
    x, inform = solve(A, b, p['radius'], A_type=A_type)
    assert inform['status'] == 0
    assert x == pytest.approx(p['x'], abs=1e-8)
    assert inform['multiplier'] == pytest.approx(p['lam'], abs=1e-8)
    assert inform['r_norm'] == pytest.approx(p['r_norm'], abs=1e-8)
    check_solution(x, inform, p['radius'], A, b)


@pytest.mark.parametrize('factorization', [
    pytest.param(1, marks=INTERIOR_XFAIL), 2])
def test_llst_factorization(terminate, factorization):
    # problem 2) with the Schur-complement (1) and augmented-system (2) SBLS
    # factorizations: the interior solution is only found with the latter
    p = PROBLEMS['interior']
    x, inform = solve(A, b, p['radius'], factorization=factorization)
    assert inform['status'] == 0
    assert x == pytest.approx(p['x'], abs=1e-8)
    assert inform['multiplier'] == pytest.approx(0.0, abs=1e-8)
    check_solution(x, inform, p['radius'], A, b)


@pytest.mark.parametrize('S_type', [
    pytest.param('coordinate', marks=SPARSE_S_XFAIL),
    pytest.param('sparse_by_rows', marks=SPARSE_S_XFAIL), 'dense',
    'diagonal'])
def test_llst_scaling(terminate, S_type):
    # problem 2) with S = diag( 1, 4 )
    x, inform = solve(A, b_s, 1.0, S=S, S_type=S_type)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_s, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(1.0, abs=1e-8)
    assert inform['r_norm'] == pytest.approx(np.sqrt(0.98), abs=1e-8)
    check_solution(x, inform, 1.0, A, b_s, S)


@pytest.mark.parametrize('S_data', [
    ('diagonal', 2, np.array([4.0, 4.0])),
    ('scaled_identity', 1, np.array([4.0]))],
    ids=['diagonal', 'scaled_identity'])
def test_llst_scaled_identity(terminate, S_data):
    # problem 2) with S = 4 I
    x, inform = solve(A, b_s, 1.0, S_data=S_data)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_4, abs=1e-8)
    assert inform['r_norm'] == pytest.approx(np.sqrt(1.52), abs=1e-8)
    check_solution(x, inform, 1.0, A, b_s, 4.0 * np.eye(2))


@pytest.mark.parametrize('S_val', [
    np.zeros(0), None],
    ids=['empty', 'None'])
def test_llst_identity(terminate, S_val):
    # problem 2) with S = I given by the identity storage scheme
    p = PROBLEMS['boundary']
    x, inform = solve(A, b, p['radius'], S_data=('identity', 0, S_val))
    assert inform['status'] == 0
    assert x == pytest.approx(p['x'], abs=1e-8)
    check_solution(x, inform, p['radius'], A, b)


def test_llst_indefinite_scaling(terminate):
    # S must be positive definite (status -15)
    with pytest.warns(RuntimeWarning):
        x, inform = solve(A, b, 1.0, S=np.diag([1.0, -1.0]),
                          S_type='diagonal')
    assert inform['status'] == -15


@pytest.mark.parametrize('radius', [0.0, -1.0], ids=['radius=0', 'radius<0'])
def test_llst_radius_not_positive(terminate, radius):
    # the radius must be positive (status -3)
    with pytest.raises(ValueError):
        solve(A, b, radius)
    assert llst.information()['status'] == -3


def test_llst_reload(terminate):
    # a new load discards the S of an earlier one, so problem 2) is solved
    # with S = I
    p = PROBLEMS['boundary']
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, 'coordinate')
    S_ne, S_row, S_col, S_ptr, S_val = s_storage(S, 'coordinate')
    llst.load(4, 2, 'coordinate', A_ne, A_row, A_col, A_ptr, options())
    llst.load_scaling(2, 'coordinate', S_ne, S_row, S_col, S_ptr)
    llst.load(4, 2, 'coordinate', A_ne, A_row, A_col, A_ptr, options())
    x = llst.solve_problem(4, 2, p['radius'], A_ne, A_val, b)
    assert llst.information()['status'] == 0
    assert x == pytest.approx(p['x'], abs=1e-8)


def test_llst_missing_s_val(terminate):
    # S_val must be given once S has been loaded (status -79, which is
    # reported only through inform)
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, 'coordinate')
    S_ne, S_row, S_col, S_ptr, S_val = s_storage(S, 'coordinate')
    llst.load(4, 2, 'coordinate', A_ne, A_row, A_col, A_ptr, options())
    llst.load_scaling(2, 'coordinate', S_ne, S_row, S_col, S_ptr)
    llst.solve_problem(4, 2, 1.0, A_ne, A_val, b)
    assert llst.information()['status'] == -79
