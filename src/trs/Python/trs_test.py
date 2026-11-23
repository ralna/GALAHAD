from galahad import trs
import numpy as np
import pytest


# trs finds the global minimizer of q(x) = f + g^T x + 1/2 x^T H x within
# the trust region ||x||_M <= radius, optionally with A x = 0, where
# ||x||_M = sqrt( x^T M x ). x is a solution if and only if, for some
# multiplier lambda >= 0 and y,
#   ( H + lambda M ) x + A^T y = - g, A x = 0,
#   lambda ( ||x||_M - radius ) = 0, ||x||_M <= radius,
# and H + lambda M is positive semi-definite on the null space of A.
#
# test problems, as in test_trs.py, with
#     ( 1 0 4 )
# H = ( 0 2 0 ), M = diag( 1, 2, 1 ), A = ( 1 1 1 ), f = 0.96 and radius 1:
#     ( 4 0 3 )
# 1) g = ( 0, 2, 0 ); 2) as 1) with M; 3) as 2) with A x = 0.
# H has eigenvalues 2 and 2 +- sqrt( 17 ), the smallest with eigenvector
# u = ( 4, 0, 1 - sqrt( 17 ) ), which is orthogonal to g. In 1) and 2) this
# is the hard case: lambda = sqrt( 17 ) - 2, minus the smallest
# eigenvalue of H (and of H with respect to M, as M is the identity on the
# span of e_0 and e_2), and ( H + lambda M ) x_1 = - g gives
# x_1 = - 2 / sqrt( 17 ) in 1) and - 2 / ( 2 + 2 lambda ) in 2), with
# ||x_1 e_1||_M < 1. The solutions x = x_1 e_1 + tau u / ||u|| (with tau
# chosen so that ||x||_M = 1, of either sign) are not unique, but lambda
# and q(x) = f + g^T x + 1/2 ( 2 x_1^2 + tau^2 ( 2 - sqrt( 17 ) ) ) are:
# 1) q = 1.96 - 21 / ( 2 sqrt( 17 ) ),
# 2) q = 0.96 + ( 15 - 9 sqrt( 17 ) ) / 16
n = 3
m = 1
f = 0.96
radius = 1.0
g = np.array([0.0, 2.0, 0.0])
H = np.array([[1.0, 0.0, 4.0], [0.0, 2.0, 0.0], [4.0, 0.0, 3.0]])
M = np.diag([1.0, 2.0, 1.0])
A = np.array([[1.0, 1.0, 1.0]])
lambda_hard = np.sqrt(17.0) - 2.0
q_hard = {False: 1.96 - 21.0 / (2.0 * np.sqrt(17.0)),
          True: 0.96 + (15.0 - 9.0 * np.sqrt(17.0)) / 16.0}

# well-posed variants, built from a chosen x and lambda with H + lambda M
# positive definite (on the null space of A) by setting
# g = - ( H + lambda M ) x - A^T y, with f = 0:
# 4) M = I, x = ( 2, -1, -2 ) / 3 with ||x|| = 1 and lambda = 3, so
# g = ( 0, 5, 4 ) / 3 and q = g^T x + 1/2 x^T H x = - 13/9 - 7/9 = - 20/9
g_b = np.array([0.0, 5.0, 4.0]) / 3.0
x_star_b = np.array([2.0, -1.0, -2.0]) / 3.0
lambda_b = 3.0
q_b = -20.0 / 9.0
# 5) M as above, x = ( 1, 1, 1 ) / 2 with ||x||_M = 1 and lambda = 3, so
# g = - ( 4, 4, 5 ) and q = - 13/2 + 7/4 = - 19/4
g_m = np.array([-4.0, -4.0, -5.0])
x_star_m = np.array([0.5, 0.5, 0.5])
lambda_m = 3.0
q_m = -19.0 / 4.0
# 6) A x = 0, with or without M, x = ( 1, 0, -1 ), radius = sqrt( 2 ),
# lambda = 3 and y = -1. As x_1 = 0, ( H + 3 M ) x = ( H + 3 I ) x =
# ( 0, 0, -2 ), so g = ( 1, 1, 3 ), and q = - 2 - 2 = - 4. The null space of
# A is spanned by ( 1, 0, -1 ) and ( 1, -2, 1 ), on which H + 3 M is
# positive definite
radius_a = np.sqrt(2.0)
g_a = np.array([1.0, 1.0, 3.0])
x_star_a = np.array([1.0, 0.0, -1.0])
y_star_a = np.array([-1.0])
lambda_a = 3.0
q_a = -4.0
# 7) the positive definite H + 3 I, with g and x as in 4), radius 2: the
# unconstrained minimizer x = ( 2, -1, -2 ) / 3 lies inside the region, so
# lambda = 0 and q = 1/2 g^T x = - 13/18
H_pd = H + 3.0 * np.eye(n)
radius_i = 2.0
q_i = -13.0 / 18.0
# 8) equality_problem, requiring ||x|| = 2, with H + 3 I, x = ( 4, -2, -4 ) / 3
# and lambda = - 1/2 (H + 5/2 I is positive definite), so
# g = - ( H + 5/2 I ) x = ( 2, 9, 6 ) / 3 and q = - 34/9 + 26/9 = - 8/9
g_e = np.array([2.0, 9.0, 6.0]) / 3.0
x_star_e = np.array([4.0, -2.0, -4.0]) / 3.0
lambda_e = -0.5
q_e = -8.0 / 9.0

H_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']
M_STORAGE = ['coordinate', 'sparse_by_rows', 'dense', 'diagonal']
A_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']
# trs uses a dense factorization (of the eigenvalues of H with respect to
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

def solve(g, radius, H=H, H_type='coordinate', M=None, M_type='coordinate',
          A=None, A_type='coordinate', f=0.0, H_data=None, M_data=None,
          **opts):
    # solve the problem, returning x, y (None without A) and inform. H_data
    # or M_data = ( type, ne, val ) give H or M directly, for the storage
    # schemes without row and column indices
    options = trs.initialize()
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
    trs.load(n, H_type, H_ne, H_row, H_col, H_ptr, options)
    args = {}
    if M_data is not None:
        M_type, M_ne, M_val = M_data
        trs.load_m(n, M_type, M_ne, None, None, None)
        args.update(M_ne=M_ne, M_val=M_val)
    elif M is not None:
        M_ne, M_row, M_col, M_ptr, M_val = sym_storage(M, M_type)
        trs.load_m(n, M_type, M_ne, M_row, M_col, M_ptr)
        args.update(M_ne=M_ne, M_val=M_val)
    if A is not None:
        A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, A_type)
        trs.load_a(m, A_type, A_ne, A_row, A_col, A_ptr)
        args.update(m=m, A_ne=A_ne, A_val=A_val)
    result = trs.solve_problem(n, radius, f, g, H_ne, H_val, **args)
    inform = trs.information()
    trs.terminate()
    if A is None:
        return result, None, inform
    x, y = result
    return x, y, inform

def check_optimality(x, y, inform, g, radius, H=H, M=None, A=None, f=0.0):
    # check the conditions above for the reported multiplier, with ||x||_M,
    # q(x) and lambda as reported in inform
    M = np.eye(n) if M is None else M
    lam = inform['multiplier']
    norm_x = np.sqrt(x @ M @ x)
    assert lam >= 0.0
    assert norm_x <= radius + 1e-8
    assert lam * (norm_x - radius) == pytest.approx(0.0, abs=1e-8)
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
    assert inform['obj'] == pytest.approx(f + g @ x + 0.5 * x @ H @ x,
                                          abs=1e-8)


@pytest.mark.parametrize('factorization', FACTORIZATION)
@pytest.mark.parametrize('H_type', H_STORAGE)
@pytest.mark.parametrize('use_m', [False, True], ids=['I', 'M'])
def test_trs_hard_case(use_m, H_type, factorization):
    # problems 1) and 2): x is not unique, so check lambda, q and the
    # optimality conditions
    Mx = M if use_m else None
    x, y, inform = solve(g, radius, H_type=H_type, M=Mx, f=f,
                         dense_factorization=factorization)
    assert inform['status'] == 0
    assert inform['hard_case']
    assert inform['multiplier'] == pytest.approx(lambda_hard, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_hard[use_m], abs=1e-8)
    assert inform['x_norm'] == pytest.approx(radius, abs=1e-8)
    check_optimality(x, y, inform, g, radius, M=Mx, f=f)


@pytest.mark.parametrize('factorization', FACTORIZATION)
@pytest.mark.parametrize('use_m', [False, True], ids=['I', 'M'])
def test_trs_example_constrained(use_m, factorization):
    # problem 3), and its variant without M: x has no closed form, so check
    # the optimality conditions
    Mx = M if use_m else None
    x, y, inform = solve(g, radius, M=Mx, A=A, f=f,
                         dense_factorization=factorization)
    assert inform['status'] == 0
    assert not inform['hard_case']
    assert inform['x_norm'] == pytest.approx(radius, abs=1e-8)
    check_optimality(x, y, inform, g, radius, M=Mx, A=A, f=f)


@pytest.mark.parametrize('factorization', FACTORIZATION)
@pytest.mark.parametrize('H_type', H_STORAGE)
def test_trs_boundary(H_type, factorization):
    # problem 4)
    x, y, inform = solve(g_b, radius, H_type=H_type,
                         dense_factorization=factorization)
    assert inform['status'] == 0
    assert not inform['hard_case']
    assert x == pytest.approx(x_star_b, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(lambda_b, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_b, abs=1e-8)
    check_optimality(x, y, inform, g_b, radius)


@pytest.mark.parametrize('factorization', FACTORIZATION)
@pytest.mark.parametrize('M_type', M_STORAGE)
def test_trs_boundary_m(M_type, factorization):
    # problem 5)
    x, y, inform = solve(g_m, radius, M=M, M_type=M_type,
                         dense_factorization=factorization)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_m, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(lambda_m, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_m, abs=1e-8)
    check_optimality(x, y, inform, g_m, radius, M=M)


@pytest.mark.parametrize('factorization', FACTORIZATION)
@pytest.mark.parametrize('A_type', A_STORAGE)
@pytest.mark.parametrize('use_m', [False, True], ids=['I', 'M'])
def test_trs_boundary_a(use_m, A_type, factorization):
    # problem 6)
    Mx = M if use_m else None
    x, y, inform = solve(g_a, radius_a, M=Mx, A=A, A_type=A_type,
                         dense_factorization=factorization)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_a, abs=1e-8)
    assert y == pytest.approx(y_star_a, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(lambda_a, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_a, abs=1e-8)
    check_optimality(x, y, inform, g_a, radius_a, M=Mx, A=A)


@pytest.mark.parametrize('factorization', FACTORIZATION)
def test_trs_interior(factorization):
    # problem 7)
    x, y, inform = solve(g_b, radius_i, H=H_pd,
                         dense_factorization=factorization)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_b, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(0.0, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_i, abs=1e-8)
    check_optimality(x, y, inform, g_b, radius_i, H=H_pd)


@pytest.mark.parametrize('factorization', FACTORIZATION)
def test_trs_equality_problem(factorization):
    # problem 8): lambda < 0 is allowed when ||x|| = radius is required
    x, y, inform = solve(g_e, radius_i, H=H_pd, equality_problem=True,
                         dense_factorization=factorization)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_e, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(lambda_e, abs=1e-8)
    assert inform['x_norm'] == pytest.approx(radius_i, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_e, abs=1e-8)
    assert (H_pd + lambda_e * np.eye(n)) @ x + g_e \
        == pytest.approx(np.zeros(n), abs=1e-8)


# 9) H = diag( 1, 2, 3 ): with g = ( 1, 1, 1 ) and radius 10, the
# minimizer x = - ( 1, 1/2, 1/3 ) of q lies inside the region, with
# lambda = 0 and q = - 11/12; with x = ( 3, 0, -4 ) / 5 and lambda = 1,
# g = - ( H + I ) x = ( -6, 0, 16 ) / 5 and q = - 82/25 + 57/50 = - 107/50
H_d = np.diag([1.0, 2.0, 3.0])
g_di = np.array([1.0, 1.0, 1.0])
x_star_di = -np.array([1.0, 0.5, 1.0 / 3.0])
q_di = -11.0 / 12.0
g_db = np.array([-6.0, 0.0, 16.0]) / 5.0
x_star_db = np.array([3.0, 0.0, -4.0]) / 5.0
q_db = -107.0 / 50.0

@pytest.mark.parametrize('factorization', FACTORIZATION)
def test_trs_diagonal_interior(factorization):
    x, y, inform = solve(g_di, 10.0, H=H_d, H_type='diagonal',
                         dense_factorization=factorization)
    assert x == pytest.approx(x_star_di, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(0.0, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_di, abs=1e-8)
    assert inform['status'] == 0


@pytest.mark.parametrize('factorization', FACTORIZATION)
def test_trs_diagonal_boundary(factorization):
    x, y, inform = solve(g_db, radius, H=H_d, H_type='diagonal',
                         dense_factorization=factorization)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_db, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(1.0, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_db, abs=1e-8)
    check_optimality(x, y, inform, g_db, radius, H=H_d)


def test_trs_radius_not_positive():
    # radius must be positive (status -3)
    options = trs.initialize()
    options['print_level'] = 0
    H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, 'coordinate')
    try:
        trs.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, options)
        with pytest.raises(ValueError):
            trs.solve_problem(n, 0.0, 0.0, g_b, H_ne, H_val)
        assert trs.information()['status'] == -3
    finally:
        trs.terminate()


def test_trs_indefinite_m():
    # M must be positive definite (status -15)
    with pytest.warns(RuntimeWarning):
        x, y, inform = solve(g_b, radius, M=np.diag([1.0, -1.0, 1.0]),
                             M_type='diagonal')
    assert inform['status'] == -15


# 10) H = s I given by the storage schemes without indices, M = I and
# g = ( 3, 0, 4 ), with ||g|| = 5 > s radius for s <= 2: x = - g / 5 lies on
# the boundary, lambda = 5 - s and q = - 5 + s / 2. With A x = 0 and
# g = ( 4, 1, -2 ), whose projection onto the null space of A is
# ( 3, 0, -3 ), x = ( -1, 0, 1 ) with ||x|| = radius = sqrt( 2 ),
# y = -1, lambda = 3 - s and q = - 6 + s
IDENTITY_XFAIL = pytest.mark.xfail(
    strict=True, reason='trs_import rejects the scaled_identity, identity, '
    'zero and none storage schemes for H (see '
    'https://github.com/ralna/GALAHAD/pull/630)')
g_s = np.array([3.0, 0.0, 4.0])
x_star_s = np.array([-0.6, 0.0, -0.8])
g_sa = np.array([4.0, 1.0, -2.0])
H_IDENTITY = [('scaled_identity', 2.0, 1, np.array([2.0])),
              ('identity', 1.0, 0, np.zeros(0)), ('identity', 1.0, 0, None),
              ('zero', 0.0, 0, np.zeros(0)), ('zero', 0.0, 0, None),
              ('none', 0.0, 0, np.zeros(0)), ('none', 0.0, 0, None)]
H_IDENTITY_IDS = ['scaled_identity', 'identity', 'identity-None', 'zero',
                  'zero-None', 'none', 'none-None']

@IDENTITY_XFAIL
@pytest.mark.parametrize('use_a', [False, True], ids=['', 'A'])
@pytest.mark.parametrize('H_type, s, H_ne, H_val', H_IDENTITY,
                         ids=H_IDENTITY_IDS)
def test_trs_h_identity(H_type, s, H_ne, H_val, use_a):
    Hs = s * np.eye(n)
    if use_a:
        x, y, inform = solve(g_sa, radius_a, A=A,
                             H_data=(H_type, H_ne, H_val))
        assert x == pytest.approx(x_star_a, abs=1e-8)
        assert y == pytest.approx(y_star_a, abs=1e-8)
        assert inform['multiplier'] == pytest.approx(3.0 - s, abs=1e-8)
        assert inform['obj'] == pytest.approx(-6.0 + s, abs=1e-8)
        check_optimality(x, y, inform, g_sa, radius_a, H=Hs, A=A)
    else:
        x, y, inform = solve(g_s, radius, H_data=(H_type, H_ne, H_val))
        assert x == pytest.approx(x_star_s, abs=1e-8)
        assert inform['multiplier'] == pytest.approx(5.0 - s, abs=1e-8)
        assert inform['obj'] == pytest.approx(-5.0 + s / 2.0, abs=1e-8)
        check_optimality(x, y, inform, g_s, radius, H=Hs)
    assert inform['status'] == 0


# 11) M = I given by the identity storage scheme, as in problems 4) and 6)
# without M. The dense factorization handles this, but the sparse one
# leaves M out of H + lambda M, and M_val = None is rejected (status -79)
M_IDENTITY_XFAIL = pytest.mark.xfail(
    strict=True, reason='trs fails with status -15 for an identity M when '
    'using a sparse factorization (see '
    'https://github.com/ralna/GALAHAD/pull/630)')

M_VAL_XFAIL = pytest.mark.xfail(
    strict=True, reason='trs requires M_val even for an identity M (see '
    'https://github.com/ralna/GALAHAD/pull/630)')

@pytest.mark.parametrize('factorization', FACTORIZATION)
@pytest.mark.parametrize('M_val', [np.zeros(0), None], ids=['', 'None'])
@pytest.mark.parametrize('use_a', [False, True], ids=['', 'A'])
def test_trs_m_identity(request, use_a, M_val, factorization):
    if M_val is None:
        request.applymarker(M_VAL_XFAIL)
    elif factorization == 0:
        request.applymarker(M_IDENTITY_XFAIL)
    if use_a:
        x, y, inform = solve(g_a, radius_a, A=A, M_data=('identity', 0, M_val),
                             dense_factorization=factorization)
        assert x == pytest.approx(x_star_a, abs=1e-8)
        assert y == pytest.approx(y_star_a, abs=1e-8)
        assert inform['multiplier'] == pytest.approx(lambda_a, abs=1e-8)
    else:
        x, y, inform = solve(g_b, radius, M_data=('identity', 0, M_val),
                             dense_factorization=factorization)
        assert x == pytest.approx(x_star_b, abs=1e-8)
        assert inform['multiplier'] == pytest.approx(lambda_b, abs=1e-8)
    assert inform['status'] == 0


def test_trs_reload():
    # a new load discards the M and A of earlier ones, so problem 4) is
    # solved without them
    options = trs.initialize()
    options['print_level'] = 0
    options['symmetric_linear_solver'] = 'sytr'
    options['definite_linear_solver'] = 'potr'
    H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, 'coordinate')
    M_ne, M_row, M_col, M_ptr, M_val = sym_storage(M, 'coordinate')
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, 'coordinate')
    try:
        trs.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, options)
        trs.load_m(n, 'coordinate', M_ne, M_row, M_col, M_ptr)
        trs.load_a(m, 'coordinate', A_ne, A_row, A_col, A_ptr)
        trs.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, options)
        x = trs.solve_problem(n, radius, 0.0, g_b, H_ne, H_val)
        assert trs.information()['status'] == 0
        assert x == pytest.approx(x_star_b, abs=1e-8)
    finally:
        trs.terminate()


def test_trs_missing_m_val():
    # M_val must be given once M has been loaded (status -79, which is
    # reported only through inform)
    options = trs.initialize()
    options['print_level'] = 0
    H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, 'coordinate')
    M_ne, M_row, M_col, M_ptr, M_val = sym_storage(M, 'coordinate')
    try:
        trs.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, options)
        trs.load_m(n, 'coordinate', M_ne, M_row, M_col, M_ptr)
        trs.solve_problem(n, radius, 0.0, g_b, H_ne, H_val)
        assert trs.information()['status'] == -79
    finally:
        trs.terminate()
