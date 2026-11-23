from galahad import llsr
import numpy as np
import pytest


# llsr finds a global minimizer of the regularized linear least-squares
# objective 1/2 ||A x - b||^2 + sigma / p ||x||_S^p, where
# ||x||_S = sqrt( x^T S x ), sigma > 0 and p >= 2, using factorizations.
# x solves the problem if ( A^T A + lambda S ) x = A^T b with
# lambda = sigma ||x||_S^( p - 2 ). inform reports lambda ('multiplier'),
# ||x||_S ('x_norm') and ||A x - b|| ('r_norm').

# 1) as in test_llsr.py: A = ( 1 1 1 ), b = 1, p = 3, sigma = 10 and S = I.
# Then x = alpha ( 1, 1, 1 ) with ( 3 + lambda ) alpha = 1 and
# lambda = 10 ||x|| = 10 sqrt( 3 ) alpha, so
# lambda ( 3 + lambda ) = 10 sqrt( 3 ), ||A x - b|| = lambda / ( 3 + lambda )
# and ||x|| = sqrt( 3 ) / ( 3 + lambda ).
# With S = diag( 1, 2, 1 ) as there, lambda has no closed form, so check the
# conditions above
A_e = np.ones((1, 3))
b_e = np.array([1.0])
S_e = np.diag([1.0, 2.0, 1.0])
lambda_e = (-3.0 + np.sqrt(9.0 + 40.0 * np.sqrt(3.0))) / 2.0

# 2) a small problem built from a chosen x and lambda:
# A = ( I ; diag( 1, 2 ) ), so that A^T A = diag( 2, 5 ), and sigma = 1:
# - S = I and b = ( 9/10, 4/5, 9/10, 2 ), so that A^T b = ( 9/5, 24/5 ):
#   x = ( 3, 4 ) / 5 has ||x|| = 1, so lambda = 1 for any p, and
#   ( A^T A + I ) x = A^T b, with ||A x - b|| = sqrt( 34 ) / 10;
# - S = diag( 1, 4 ) and b = ( 9/10, 6/5, 9/10, 6/5 ), so that
#   A^T b = ( 9/5, 18/5 ): x = ( 3, 2 ) / 5 has ||x||_S = 1, lambda = 1 and
#   ( A^T A + S ) x = A^T b, with ||A x - b||^2 = 98/100;
# - S = 4 I and the same b: x = ( 3, 4 ) / 10 has ||x||_S = 1, lambda = 1
#   and ( A^T A + 4 I ) x = A^T b, with ||A x - b||^2 = 152/100
A = np.vstack([np.eye(2), np.diag([1.0, 2.0])])
b = np.array([0.9, 0.8, 0.9, 2.0])
x_star = np.array([0.6, 0.8])
b_s = np.array([0.9, 1.2, 0.9, 1.2])
S = np.diag([1.0, 4.0])
x_star_s = np.array([0.6, 0.4])
x_star_4 = np.array([0.3, 0.4])

A_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']
S_STORAGE = ['coordinate', 'sparse_by_rows', 'dense', 'diagonal']

SCALING_XFAIL = 'https://github.com/ralna/GALAHAD/pull/642'

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

def options():
    o = llsr.initialize()
    o['print_level'] = 0
    # use the LAPACK solvers sytr and potr rather than the defaults, which
    # need the environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    o['definite_linear_solver'] = 'potr'
    o['sbls_options']['symmetric_linear_solver'] = 'sytr'
    o['sbls_options']['definite_linear_solver'] = 'potr'
    return o

def solve(A, b, power, weight, A_type='coordinate', S=None,
          S_type='coordinate', S_data=None):
    # solve the problem, returning x and inform. S_data = ( type, ne, val )
    # gives S directly, for the storage schemes without indices
    m, n = A.shape
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, A_type)
    llsr.load(m, n, A_type, A_ne, A_row, A_col, A_ptr, options())
    args = {}
    if S_data is not None:
        S_type, S_ne, S_val = S_data
        llsr.load_scaling(n, S_type, S_ne, None, None, None)
        args = dict(S_ne=S_ne, S_val=S_val)
    elif S is not None:
        S_ne, S_row, S_col, S_ptr, S_val = s_storage(S, S_type)
        llsr.load_scaling(n, S_type, S_ne, S_row, S_col, S_ptr)
        args = dict(S_ne=S_ne, S_val=S_val)
    x = llsr.solve_problem(m, n, power, weight, A_ne, A_val, b, **args)
    return x, llsr.information()

def check_solution(x, inform, power, weight, A, b, S=None):
    # check the conditions above for the reported multiplier, and the
    # reported norms
    S = np.eye(len(x)) if S is None else S
    lam = inform['multiplier']
    norm_x = np.sqrt(x @ S @ x)
    assert lam == pytest.approx(weight * norm_x ** (power - 2.0), abs=1e-8)
    assert (A.T @ A + lam * S) @ x == pytest.approx(A.T @ b, abs=1e-8)
    assert inform['x_norm'] == pytest.approx(norm_x, abs=1e-8)
    assert inform['r_norm'] == pytest.approx(np.linalg.norm(A @ x - b),
                                             abs=1e-8)


@pytest.fixture
def terminate():
    yield
    llsr.terminate()


def test_llsr_example(terminate):
    # problem 1) with S = I
    x, inform = solve(A_e, b_e, 3.0, 10.0)
    assert inform['status'] == 0
    assert x == pytest.approx(np.ones(3) / (3.0 + lambda_e), abs=1e-8)
    assert inform['multiplier'] == pytest.approx(lambda_e, abs=1e-8)
    assert inform['r_norm'] == pytest.approx(lambda_e / (3.0 + lambda_e),
                                             abs=1e-8)
    check_solution(x, inform, 3.0, 10.0, A_e, b_e)


def test_llsr_example_scaled(terminate):
    # problem 1) with S = diag( 1, 2, 1 )
    x, inform = solve(A_e, b_e, 3.0, 10.0, S=S_e)
    assert inform['status'] == 0
    check_solution(x, inform, 3.0, 10.0, A_e, b_e, S_e)


@pytest.mark.parametrize('A_type', A_STORAGE)
@pytest.mark.parametrize('power', [2.0, 3.0, 4.0],
                         ids=['p=2', 'p=3', 'p=4'])
def test_llsr(terminate, power, A_type):
    # problem 2) with S = I
    x, inform = solve(A, b, power, 1.0, A_type=A_type)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(1.0, abs=1e-8)
    assert inform['r_norm'] == pytest.approx(np.sqrt(34.0) / 10.0,
                                             abs=1e-8)
    check_solution(x, inform, power, 1.0, A, b)


@pytest.mark.parametrize('S_type', S_STORAGE)
def test_llsr_scaling(terminate, S_type):
    # problem 2) with S = diag( 1, 4 )
    x, inform = solve(A, b_s, 3.0, 1.0, S=S, S_type=S_type)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_s, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(1.0, abs=1e-8)
    assert inform['r_norm'] == pytest.approx(np.sqrt(0.98), abs=1e-8)
    check_solution(x, inform, 3.0, 1.0, A, b_s, S)


@pytest.mark.parametrize('S_data', [
    ('diagonal', 2, np.array([4.0, 4.0])),
    pytest.param(('scaled_identity', 1, np.array([4.0])),
                 marks=pytest.mark.xfail(
                     strict=True, reason='llsr rejects a scaled-identity S '
                     f'(see {SCALING_XFAIL})'))],
    ids=['diagonal', 'scaled_identity'])
def test_llsr_scaled_identity(terminate, S_data):
    # problem 2) with S = 4 I
    x, inform = solve(A, b_s, 3.0, 1.0, S_data=S_data)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_4, abs=1e-8)
    assert inform['r_norm'] == pytest.approx(np.sqrt(1.52), abs=1e-8)
    check_solution(x, inform, 3.0, 1.0, A, b_s, 4.0 * np.eye(2))


@pytest.mark.parametrize('S_val', [
    np.zeros(0),
    pytest.param(None, marks=pytest.mark.xfail(
        strict=True, reason='llsr requires S_val for an identity S (see '
        f'{SCALING_XFAIL})'))],
    ids=['empty', 'None'])
def test_llsr_identity(terminate, S_val):
    # problem 2) with S = I given by the identity storage scheme
    x, inform = solve(A, b, 3.0, 1.0, S_data=('identity', 0, S_val))
    assert inform['status'] == 0
    assert x == pytest.approx(x_star, abs=1e-8)
    check_solution(x, inform, 3.0, 1.0, A, b)


@pytest.mark.xfail(strict=True, reason='llsr does not check that a diagonal '
                   f'S is positive definite (see {SCALING_XFAIL})')
def test_llsr_indefinite_scaling(terminate):
    # S must be positive definite (status -15)
    with pytest.warns(RuntimeWarning):
        x, inform = solve(A, b, 3.0, 1.0, S=np.diag([1.0, -1.0]),
                          S_type='diagonal')
    assert inform['status'] == -15


@pytest.mark.parametrize('power, weight', [(3.0, 0.0), (3.0, -1.0),
                                           (1.5, 1.0)],
                         ids=['sigma=0', 'sigma<0', 'p<2'])
def test_llsr_bad_parameters(terminate, power, weight):
    # sigma must be positive and p >= 2 (status -3)
    with pytest.raises(ValueError):
        solve(A, b, power, weight)
    assert llsr.information()['status'] == -3


def test_llsr_reload(terminate):
    # a new load discards the S of an earlier one, so problem 2) is solved
    # with S = I
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, 'coordinate')
    S_ne, S_row, S_col, S_ptr, S_val = s_storage(S, 'coordinate')
    llsr.load(4, 2, 'coordinate', A_ne, A_row, A_col, A_ptr, options())
    llsr.load_scaling(2, 'coordinate', S_ne, S_row, S_col, S_ptr)
    llsr.load(4, 2, 'coordinate', A_ne, A_row, A_col, A_ptr, options())
    x = llsr.solve_problem(4, 2, 3.0, 1.0, A_ne, A_val, b)
    assert llsr.information()['status'] == 0
    assert x == pytest.approx(x_star, abs=1e-8)


def test_llsr_missing_s_val(terminate):
    # S_val must be given once S has been loaded (status -79, which is
    # reported only through inform)
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, 'coordinate')
    S_ne, S_row, S_col, S_ptr, S_val = s_storage(S, 'coordinate')
    llsr.load(4, 2, 'coordinate', A_ne, A_row, A_col, A_ptr, options())
    llsr.load_scaling(2, 'coordinate', S_ne, S_row, S_col, S_ptr)
    llsr.solve_problem(4, 2, 3.0, 1.0, A_ne, A_val, b)
    assert llsr.information()['status'] == -79
