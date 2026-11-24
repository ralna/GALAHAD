from galahad import rqs
import numpy as np
import pytest


# rqs finds the global minimizer of the regularized quadratic
# r(x) = q(x) + sigma / p ||x||_M^p, where q(x) = f + g^T x + 1/2 x^T H x,
# ||x||_M = sqrt( x^T M x ), p >= 2 and sigma > 0, optionally with A x = 0.
# x is a solution if and only if, for some y and
# lambda = sigma ||x||_M^( p - 2 ),
#   ( H + lambda M ) x + A^T y = - g, A x = 0,
# and H + lambda M is positive semi-definite on the null space of A.
# inform['obj'] reports q(x) and inform['obj_regularized'] r(x).
#
# test problems, as in test_rqs.py, with
#     ( 1 0 4 )
# H = ( 0 2 0 ), M = diag( 1, 2, 1 ), A = ( 1 1 1 ), f = 0.96, p = 3 and
#     ( 4 0 3 )
# sigma = 1, so that lambda = ||x||_M:
# 1) g = ( 0, 2, 0 ); 2) as 1) with M; 3) as 2) with A x = 0.
# H has eigenvalues 2 and 2 +- sqrt( 17 ), the smallest with eigenvector
# u = ( 4, 0, 1 - sqrt( 17 ) ), which is orthogonal to g. In 1) and 2) this
# is the hard case: lambda = sqrt( 17 ) - 2, minus the smallest
# eigenvalue of H (and of H with respect to M, as M is the identity on the
# span of e_0 and e_2), and ( H + lambda M ) x_1 = - g gives
# x_1 = - 2 / sqrt( 17 ) in 1) and - 2 / ( 2 + 2 lambda ) in 2), with
# ||x_1 e_1||_M < lambda. The solutions x = x_1 e_1 + tau u / ||u|| (with
# tau chosen so that ||x||_M = lambda, of either sign) are not unique, but
# lambda and q(x) = f + 2 x_1 + 1/2 ( 2 x_1^2 - tau^2 lambda ) are, as
# x^T H x = 2 x_1^2 + tau^2 ( 2 - sqrt( 17 ) ), with
# tau^2 = lambda^2 - ||x_1 e_1||_M^2, and r(x) = q(x) + lambda^3 / 3
n = 3
m = 1
f = 0.96
power = 3.0
weight = 1.0
g = np.array([0.0, 2.0, 0.0])
H = np.array([[1.0, 0.0, 4.0], [0.0, 2.0, 0.0], [4.0, 0.0, 3.0]])
M = np.diag([1.0, 2.0, 1.0])
A = np.array([[1.0, 1.0, 1.0]])
lambda_hard = np.sqrt(17.0) - 2.0

def hard_case_q(use_m):
    if use_m:
        x_1 = - 2.0 / (2.0 + 2.0 * lambda_hard)
        tau2 = lambda_hard ** 2 - 2.0 * x_1 ** 2
    else:
        x_1 = - 2.0 / np.sqrt(17.0)
        tau2 = lambda_hard ** 2 - x_1 ** 2
    return f + 2.0 * x_1 + 0.5 * (2.0 * x_1 ** 2 - tau2 * lambda_hard)

# well-posed variants, built from a chosen x and lambda with H + lambda M
# positive definite (on the null space of A) by setting
# g = - ( H + lambda M ) x - A^T y, with f = 0:
# 4) M = I, x = ( 2, -1, -2 ) with ||x|| = 3 and lambda = 3, so
# g = ( 0, 5, 4 ) and q = - 13 - 7 = - 20. lambda = sigma ||x||^( p - 2 )
# = 3 for p = 2, 3 and 4 with sigma = 3, 1 and 1/3, when
# r = q + sigma / p 3^p = - 13/2, - 11 and - 53/4
g_b = np.array([0.0, 5.0, 4.0])
x_star_b = np.array([2.0, -1.0, -2.0])
lambda_b = 3.0
q_b = -20.0
POWERS = [(2.0, 3.0, -13.0 / 2.0), (3.0, 1.0, -11.0),
          (4.0, 1.0 / 3.0, -53.0 / 4.0)]
POWER_IDS = ['p=2', 'p=3', 'p=4']
# 5) M as above, p = 3, sigma = 1, x = ( 3, 3, 3 ) / 2 with ||x||_M = 3 and
# lambda = 3, so g = - ( 12, 12, 15 ), q = - 117/2 + 63/4 = - 171/4 and
# r = q + 9 = - 135/4
g_m = np.array([-12.0, -12.0, -15.0])
x_star_m = np.array([1.5, 1.5, 1.5])
lambda_m = 3.0
q_m = -171.0 / 4.0
r_m = -135.0 / 4.0
# 6) A x = 0, with or without M, y = -1 and lambda = 3: with p = 2 and
# sigma = 3, x = ( 1, 0, -1 ) for any ||x||, and as x_1 = 0,
# ( H + 3 M ) x = ( H + 3 I ) x = ( 0, 0, -2 ), so g = ( 1, 1, 3 ),
# q = - 2 - 2 = - 4 and r = q + 3 = - 1. With p = 3 and sigma = 1,
# x = k ( 1, 0, -1 ) with k = 3 / sqrt( 2 ), so that ||x|| = 3, giving
# g = ( 1, 1, 1 + 2 k ), q = - 2 k^2 - 2 k^2 = - 18 and r = q + 9 = - 9.
# The null space of A is spanned by ( 1, 0, -1 ) and ( 1, -2, 1 ), on which
# H + 3 M is positive definite
k_a = 3.0 / np.sqrt(2.0)
A_PROBLEMS = {
    'p=2': dict(power=2.0, weight=3.0, g=np.array([1.0, 1.0, 3.0]),
                x=np.array([1.0, 0.0, -1.0]), q=-4.0, r=-1.0),
    'p=3': dict(power=3.0, weight=1.0, g=np.array([1.0, 1.0, 1.0 + 2.0 * k_a]),
                x=k_a * np.array([1.0, 0.0, -1.0]), q=-18.0, r=-9.0)}
y_star_a = np.array([-1.0])
lambda_a = 3.0
# 7) H = diag( 1, 2, 3 ), x = ( 2, -1, -2 ) / 3 with ||x|| = 1 and
# lambda = 1/2, which is sigma ||x||^( p - 2 ) for sigma = 1/2 and any p,
# so g = - ( H + I / 2 ) x = ( -6, 5, 14 ) / 6, q = - 5/2 + 1 = - 3/2 and
# r = q + 1 / ( 2 p ) = - 5/4 for p = 2 and - 4/3 for p = 3
H_d = np.diag([1.0, 2.0, 3.0])
g_d = np.array([-6.0, 5.0, 14.0]) / 6.0
x_star_d = np.array([2.0, -1.0, -2.0]) / 3.0
q_d = -1.5
r_d = {2.0: -1.25, 3.0: -4.0 / 3.0}

H_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']
M_STORAGE = ['coordinate', 'sparse_by_rows', 'dense', 'diagonal']
A_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']
# rqs uses a dense factorization (of the eigenvalues of H with respect to
# M) or a sparse one (of H + lambda M) as options['dense_factorization']
# is 1 or 0
FACTORIZATION = [0, 1]

def sym_storage(H, H_type):
    # return H_ne, H_row, H_col, H_ptr, H_val for the lower triangle of H
    if H_type == 'dense':
        rows, cols = np.tril_indices(H.shape[0])
        return len(rows), None, None, None, H[rows, cols]
    if H_type == 'diagonal':
        return H.shape[0], None, None, None, np.diag(H).copy()
    rows, cols = np.nonzero(np.tril(H))
    if H_type == 'coordinate':
        return len(rows), rows, cols, None, H[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=H.shape[0]))]
    return len(rows), None, cols, ptr, H[rows, cols]

def a_storage(A, A_type):
    # return A_ne, A_row, A_col, A_ptr, A_val for A
    if A_type == 'dense':
        return A.size, None, None, None, A.ravel()
    rows, cols = np.nonzero(A)
    if A_type == 'coordinate':
        return len(rows), rows, cols, None, A[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=A.shape[0]))]
    return len(rows), None, cols, ptr, A[rows, cols]

def solve(g, power, weight, H=H, H_type='coordinate', M=None,
          M_type='coordinate', A=None, A_type='coordinate', f=0.0,
          H_data=None, M_data=None, **opts):
    # solve the problem, returning x, y (None without A) and inform. H_data
    # or M_data = ( type, ne, val ) give H or M directly, for the storage
    # schemes without row and column indices
    options = rqs.initialize()
    options['print_level'] = 0
    # use the LAPACK solvers sytr and potr rather than the defaults, which
    # need the environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    options['symmetric_linear_solver'] = 'sytr'
    options['definite_linear_solver'] = 'potr'
    options.update(opts)
    if H_data is None:
        H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, H_type)
    else:
        (H_type, H_ne, H_val), H_row, H_col, H_ptr = H_data, None, None, None
    rqs.load(n, H_type, H_ne, H_row, H_col, H_ptr, options)
    args = {}
    if M_data is not None:
        M_type, M_ne, M_val = M_data
        rqs.load_m(n, M_type, M_ne, None, None, None)
        args.update(M_ne=M_ne, M_val=M_val)
    elif M is not None:
        M_ne, M_row, M_col, M_ptr, M_val = sym_storage(M, M_type)
        rqs.load_m(n, M_type, M_ne, M_row, M_col, M_ptr)
        args.update(M_ne=M_ne, M_val=M_val)
    if A is not None:
        A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, A_type)
        rqs.load_a(m, A_type, A_ne, A_row, A_col, A_ptr)
        args.update(m=m, A_ne=A_ne, A_val=A_val)
    result = rqs.solve_problem(n, power, weight, f, g, H_ne, H_val, **args)
    inform = rqs.information()
    rqs.terminate()
    if A is None:
        return result, None, inform
    x, y = result
    return x, y, inform

def check_optimality(x, y, inform, g, power, weight, H=H, M=None, A=None,
                     f=0.0):
    # check the conditions above, with ||x||_M, lambda, q(x) and r(x) as
    # reported in inform
    M = np.eye(n) if M is None else M
    lam = inform['multiplier']
    norm_x = np.sqrt(x @ M @ x)
    assert lam == pytest.approx(weight * norm_x ** (power - 2.0), abs=1e-8)
    grad = (H + lam * M) @ x + g
    if A is not None:
        assert A @ x == pytest.approx(np.zeros(m), abs=1e-8)
        grad += A.T @ y
        Z = np.linalg.svd(A)[2][A.shape[0]:].T  # null-space basis
        assert np.linalg.eigvalsh(Z.T @ (H + lam * M) @ Z).min() >= -1e-8
    else:
        assert np.linalg.eigvalsh(H + lam * M).min() >= -1e-8
    assert grad == pytest.approx(np.zeros(n), abs=1e-8)
    assert inform['x_norm'] == pytest.approx(norm_x, abs=1e-8)
    q = f + g @ x + 0.5 * x @ H @ x
    assert inform['obj'] == pytest.approx(q, abs=1e-8)
    assert inform['obj_regularized'] \
        == pytest.approx(q + weight / power * norm_x ** power, abs=1e-8)


@pytest.mark.parametrize('factorization', FACTORIZATION)
@pytest.mark.parametrize('H_type', H_STORAGE)
@pytest.mark.parametrize('use_m', [False, True], ids=['I', 'M'])
def test_rqs_hard_case(use_m, H_type, factorization):
    # problems 1) and 2): x is not unique, so check lambda, q, r and the
    # optimality conditions
    Mx = M if use_m else None
    x, y, inform = solve(g, power, weight, H_type=H_type, M=Mx, f=f,
                         dense_factorization=factorization)
    assert inform['status'] == 0
    assert inform['hard_case']
    assert inform['multiplier'] == pytest.approx(lambda_hard, abs=1e-8)
    assert inform['x_norm'] == pytest.approx(lambda_hard, abs=1e-8)
    q = hard_case_q(use_m)
    assert inform['obj'] == pytest.approx(q, abs=1e-8)
    assert inform['obj_regularized'] \
        == pytest.approx(q + lambda_hard ** 3 / 3.0, abs=1e-8)
    check_optimality(x, y, inform, g, power, weight, M=Mx, f=f)


@pytest.mark.parametrize('factorization', FACTORIZATION)
@pytest.mark.parametrize('use_m', [False, True], ids=['I', 'M'])
def test_rqs_example_constrained(use_m, factorization):
    # problem 3), and its variant without M: x has no closed form, so check
    # the optimality conditions
    Mx = M if use_m else None
    x, y, inform = solve(g, power, weight, M=Mx, A=A, f=f,
                         dense_factorization=factorization)
    assert inform['status'] == 0
    assert not inform['hard_case']
    check_optimality(x, y, inform, g, power, weight, M=Mx, A=A, f=f)


@pytest.mark.parametrize('factorization', FACTORIZATION)
@pytest.mark.parametrize('H_type', H_STORAGE)
@pytest.mark.parametrize('power, weight, r_b', POWERS, ids=POWER_IDS)
def test_rqs_powers(power, weight, r_b, H_type, factorization):
    # problem 4)
    x, y, inform = solve(g_b, power, weight, H_type=H_type,
                         dense_factorization=factorization)
    assert inform['status'] == 0
    assert not inform['hard_case']
    assert x == pytest.approx(x_star_b, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(lambda_b, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_b, abs=1e-8)
    assert inform['obj_regularized'] == pytest.approx(r_b, abs=1e-8)
    check_optimality(x, y, inform, g_b, power, weight)


@pytest.mark.parametrize('factorization', FACTORIZATION)
@pytest.mark.parametrize('M_type', M_STORAGE)
def test_rqs_m(M_type, factorization):
    # problem 5)
    x, y, inform = solve(g_m, power, weight, M=M, M_type=M_type,
                         dense_factorization=factorization)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_m, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(lambda_m, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_m, abs=1e-8)
    assert inform['obj_regularized'] == pytest.approx(r_m, abs=1e-8)
    check_optimality(x, y, inform, g_m, power, weight, M=M)


@pytest.mark.parametrize('factorization', FACTORIZATION)
@pytest.mark.parametrize('A_type', A_STORAGE)
@pytest.mark.parametrize('use_m', [False, True], ids=['I', 'M'])
@pytest.mark.parametrize('problem', A_PROBLEMS)
def test_rqs_a(problem, use_m, A_type, factorization):
    # problem 6)
    p = A_PROBLEMS[problem]
    Mx = M if use_m else None
    x, y, inform = solve(p['g'], p['power'], p['weight'], M=Mx, A=A,
                         A_type=A_type, dense_factorization=factorization)
    assert inform['status'] == 0
    assert x == pytest.approx(p['x'], abs=1e-8)
    assert y == pytest.approx(y_star_a, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(lambda_a, abs=1e-8)
    assert inform['obj'] == pytest.approx(p['q'], abs=1e-8)
    assert inform['obj_regularized'] == pytest.approx(p['r'], abs=1e-8)
    check_optimality(x, y, inform, p['g'], p['power'], p['weight'], M=Mx,
                     A=A)


@pytest.mark.parametrize('factorization', FACTORIZATION)
@pytest.mark.parametrize('power', [2.0, 3.0], ids=['p=2', 'p=3'])
def test_rqs_diagonal(power, factorization):
    # problem 7)
    x, y, inform = solve(g_d, power, 0.5, H=H_d, H_type='diagonal',
                         dense_factorization=factorization)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_d, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(0.5, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_d, abs=1e-8)
    assert inform['obj_regularized'] == pytest.approx(r_d[power], abs=1e-8)
    check_optimality(x, y, inform, g_d, power, 0.5, H=H_d)


@pytest.mark.parametrize('power, weight', [(1.5, 1.0), (3.0, 0.0),
                                           (3.0, -1.0)],
                         ids=['p<2', 'sigma=0', 'sigma<0'])
def test_rqs_bad_parameters(power, weight):
    # p >= 2 and sigma > 0 are required (status -3)
    options = rqs.initialize()
    options['print_level'] = 0
    H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, 'coordinate')
    try:
        rqs.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, options)
        with pytest.raises(ValueError):
            rqs.solve_problem(n, power, weight, 0.0, g_b, H_ne, H_val)
        assert rqs.information()['status'] == -3
    finally:
        rqs.terminate()


def test_rqs_indefinite_m():
    # M must be positive definite (status -15)
    with pytest.warns(RuntimeWarning):
        x, y, inform = solve(g_b, power, weight, M=np.diag([1.0, -1.0, 1.0]),
                             M_type='diagonal')
    assert inform['status'] == -15


# 8) H = s I given by the storage schemes without indices, M = I, p = 2
# and sigma = 1, so that lambda = 1 and ( s + 1 ) x = - g - A^T y. With
# g = ( 3, 0, 4 ), x = - g / ( s + 1 ), q = - 25 ( s + 2 ) / ( 2 ( s + 1 )^2 )
# and r = q + 25 / ( 2 ( s + 1 )^2 ) = - 25 / ( 2 ( s + 1 ) ). With
# A x = 0 and g = ( 4, 1, -2 ), whose projection onto the null space of A
# is ( 3, 0, -3 ), x = - ( 3, 0, -3 ) / ( s + 1 ) and y = -1, with
# q = - 9 ( s + 2 ) / ( s + 1 )^2 and r = - 9 / ( s + 1 )
g_s = np.array([3.0, 0.0, 4.0])
g_sa = np.array([4.0, 1.0, -2.0])
H_IDENTITY = [('scaled_identity', 2.0, 1, np.array([2.0])),
              ('identity', 1.0, 0, np.zeros(0)), ('identity', 1.0, 0, None),
              ('zero', 0.0, 0, np.zeros(0)), ('zero', 0.0, 0, None),
              ('none', 0.0, 0, np.zeros(0)), ('none', 0.0, 0, None)]
H_IDENTITY_IDS = ['scaled_identity', 'identity', 'identity-None', 'zero',
                  'zero-None', 'none', 'none-None']

@pytest.mark.parametrize('use_a', [False, True], ids=['', 'A'])
@pytest.mark.parametrize('H_type, s, H_ne, H_val', H_IDENTITY,
                         ids=H_IDENTITY_IDS)
def test_rqs_h_identity(H_type, s, H_ne, H_val, use_a):
    Hs = s * np.eye(n)
    if use_a:
        x, y, inform = solve(g_sa, 2.0, 1.0, A=A,
                             H_data=(H_type, H_ne, H_val))
        assert x == pytest.approx(-np.array([3.0, 0.0, -3.0]) / (s + 1.0),
                                  abs=1e-8)
        assert y == pytest.approx(y_star_a, abs=1e-8)
        assert inform['obj'] \
            == pytest.approx(-9.0 * (s + 2.0) / (s + 1.0) ** 2, abs=1e-8)
        assert inform['obj_regularized'] \
            == pytest.approx(-9.0 / (s + 1.0), abs=1e-8)
        check_optimality(x, y, inform, g_sa, 2.0, 1.0, H=Hs, A=A)
    else:
        x, y, inform = solve(g_s, 2.0, 1.0, H_data=(H_type, H_ne, H_val))
        assert x == pytest.approx(-g_s / (s + 1.0), abs=1e-8)
        assert inform['obj'] == pytest.approx(
            -25.0 * (s + 2.0) / (2.0 * (s + 1.0) ** 2), abs=1e-8)
        assert inform['obj_regularized'] \
            == pytest.approx(-25.0 / (2.0 * (s + 1.0)), abs=1e-8)
        check_optimality(x, y, inform, g_s, 2.0, 1.0, H=Hs)
    assert inform['multiplier'] == pytest.approx(1.0, abs=1e-8)
    assert inform['status'] == 0


# 9) M = I given by the identity storage scheme, with M_val empty or
# omitted, as in problems 4) (p = 3) and 6) (p = 2) without M

@pytest.mark.parametrize('factorization', FACTORIZATION)
@pytest.mark.parametrize('M_val', [np.zeros(0), None], ids=['', 'None'])
@pytest.mark.parametrize('use_a', [False, True], ids=['', 'A'])
def test_rqs_m_identity(use_a, M_val, factorization):
    if use_a:
        p = A_PROBLEMS['p=2']
        x, y, inform = solve(p['g'], p['power'], p['weight'], A=A,
                             M_data=('identity', 0, M_val),
                             dense_factorization=factorization)
        assert x == pytest.approx(p['x'], abs=1e-8)
        assert y == pytest.approx(y_star_a, abs=1e-8)
        assert inform['multiplier'] == pytest.approx(lambda_a, abs=1e-8)
    else:
        x, y, inform = solve(g_b, power, weight,
                             M_data=('identity', 0, M_val),
                             dense_factorization=factorization)
        assert x == pytest.approx(x_star_b, abs=1e-8)
        assert inform['multiplier'] == pytest.approx(lambda_b, abs=1e-8)
    assert inform['status'] == 0


def test_rqs_reload():
    # a new load discards the M and A of earlier ones, so problem 4) is
    # solved without them
    options = rqs.initialize()
    options['print_level'] = 0
    options['symmetric_linear_solver'] = 'sytr'
    options['definite_linear_solver'] = 'potr'
    H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, 'coordinate')
    M_ne, M_row, M_col, M_ptr, M_val = sym_storage(M, 'coordinate')
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, 'coordinate')
    try:
        rqs.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, options)
        rqs.load_m(n, 'coordinate', M_ne, M_row, M_col, M_ptr)
        rqs.load_a(m, 'coordinate', A_ne, A_row, A_col, A_ptr)
        rqs.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, options)
        x = rqs.solve_problem(n, power, weight, 0.0, g_b, H_ne, H_val)
        assert rqs.information()['status'] == 0
        assert x == pytest.approx(x_star_b, abs=1e-8)
    finally:
        rqs.terminate()


def test_rqs_missing_m_val():
    # M_val must be given once M has been loaded (status -79, which is
    # reported only through inform)
    options = rqs.initialize()
    options['print_level'] = 0
    H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, 'coordinate')
    M_ne, M_row, M_col, M_ptr, M_val = sym_storage(M, 'coordinate')
    try:
        rqs.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, options)
        rqs.load_m(n, 'coordinate', M_ne, M_row, M_col, M_ptr)
        rqs.solve_problem(n, power, weight, 0.0, g_b, H_ne, H_val)
        assert rqs.information()['status'] == -79
    finally:
        rqs.terminate()
