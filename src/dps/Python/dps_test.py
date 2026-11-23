from galahad import dps
import numpy as np
import pytest


# dps minimizes q(x) = f + g^T x + 1/2 x^T H x within the trust region
# ||x||_M <= radius (solve_tr_problem), or the regularized
# r(x) = q(x) + sigma / p ||x||_M^p (solve_rq_problem), in a norm
# ||x||_M = sqrt( x^T M x ) that it builds from a factorization
# H = P L D L^T P^T: M = P L |D| L^T P^T, where |D| has the eigenvalues of
# each block of D replaced by their absolute values (or by eigen_min if they
# are smaller), or M = P L L^T P^T for the goldfarb variant. x is a solution
# if, for some lambda >= 0, ( H + lambda M ) x = - g, with
# lambda ( ||x||_M - radius ) = 0 and ||x||_M <= radius for the trust region,
# and lambda = sigma ||x||_M^( p - 2 ) for the regularization, and
# H + lambda M is positive semi-definite. inform['obj'] reports q(x),
# inform['obj_regularized'] r(x) and inform['x_norm'] ||x||_M.
#
# M depends on the pivoting, and so on the linear solver, but in some
# cases it is known:
# * if H is positive definite, D is, and M = H;
# * if H is diagonal, L = I and M = |H| (or I for the goldfarb variant).
n = 3

# 1) as in test_dps.py:
#     ( 1 0 4 )
# H = ( 0 2 0 ), g = ( 0, 2, 0 ), f = 0.96, radius 1, sigma = 1 and p = 3.
#     ( 4 0 3 )
# Whatever the pivoting, H is block diagonal, with the block 2 for x_1 and
# an indefinite block for ( x_0, x_2 ), and g only involves x_1. In the
# variables z = |D|^1/2 L^T P^T x, ||x||_M = ||z|| and
# q = f + gz^T z + 1/2 ( z_1^2 + z_a^2 - z_b^2 ), where gz = ( 0, sqrt( 2 ),
# 0 ) and z_a, z_b correspond to the indefinite block. This is the hard
# case: lambda = 1, z_1 = - sqrt( 2 ) / 2 and z_b^2 = 1 - z_1^2 = 1/2
# (z_a = 0), so q = 0.96 - 1 + 1/2 ( 1/2 - 1/2 ) = - 0.04. The same z
# solves the regularized problem, as lambda = ||z|| = 1, with r = q + 1/3.
# x itself is not unique and M depends on the pivoting, so only lambda, q
# and r are checked
f = 0.96
g = np.array([0.0, 2.0, 0.0])
H = np.array([[1.0, 0.0, 4.0], [0.0, 2.0, 0.0], [4.0, 0.0, 3.0]])
q_hard = -0.04
r_hard = -0.04 + 1.0 / 3.0

# 2) the positive definite H + 3 I, so that M = H + 3 I, with
# g = ( 0, 5, 4 ) / 3 and f = 0. The unconstrained minimizer is
# x_n = ( 2, -1, -2 ) / 3, with ||x_n||_M^2 = - g^T x_n = 13/9, and any
# solution is x = t x_n with ( 1 + lambda ) t = 1, so that
# q = - 13/9 ( t - t^2 / 2 ):
# - radius 2 > ||x_n||_M: t = 1, lambda = 0 and q = - 13/18;
# - radius 1/2: t = radius / ||x_n||_M = 3 / ( 2 sqrt( 13 ) );
# - p = 2, sigma = 1: lambda = 1, t = 1/2, q = - 13/24 and
#   r = q + 1/2 ||x||_M^2 = - 13/36;
# - p = 3, sigma = 1: lambda = ||x||_M = t sqrt( 13 ) / 3, so
#   lambda^2 + lambda - sqrt( 13 ) / 3 = 0
H_pd = H + 3.0 * np.eye(n)
g_pd = np.array([0.0, 5.0, 4.0]) / 3.0
x_n = np.array([2.0, -1.0, -2.0]) / 3.0
norm_n = np.sqrt(13.0) / 3.0

def pd_values(t):
    # return x, q and lambda for x = t x_n
    return t * x_n, -13.0 / 9.0 * (t - t ** 2 / 2.0), 1.0 / t - 1.0

def lambda_p3(sigma):
    # lambda for p = 3, the root of lambda ( 1 + lambda ) = sigma ||x_n||_M
    return (-1.0 + np.sqrt(1.0 + 4.0 * sigma * norm_n)) / 2.0

P2_XFAIL = pytest.mark.xfail(
    strict=True, reason='dps rejects the regularization power p = 2 (see '
    'https://github.com/ralna/GALAHAD/pull/633)')
PD_TR = [(2.0, 1.0), (0.5, 0.5 / norm_n)]
PD_TR_IDS = ['interior', 'boundary']
PD_RQ = [pytest.param(2.0, 1.0, 0.5, marks=P2_XFAIL),
         (3.0, 1.0, 1.0 / (1.0 + lambda_p3(1.0)))]
PD_RQ_IDS = ['p=2', 'p=3']

# resolve_tr_problem and resolve_rq_problem reuse the factorization for
# new g, radius or sigma. With g / 2 the solutions are x = t x_n / 2 with
# ( 1 + lambda ) t = 1, so that q = - 13/36 ( t - t^2 / 2 )

# 3) the indefinite diagonal H = diag( 1, -2, 3 ), so that
# M = diag( 1, 2, 3 ): x = ( 1, 0, 1 ) / 2 has ||x||_M = 1, and with
# lambda = 2, H + 2 M = diag( 3, 2, 9 ) is positive definite and
# g = - ( H + 2 M ) x = - ( 3, 0, 9 ) / 2, with q = - 3 + 1/2 = - 5/2. x is
# also the regularized solution for p = 3 and sigma = 2, with
# r = q + 2/3 = - 11/6
H_d = np.diag([1.0, -2.0, 3.0])
M_d = np.diag([1.0, 2.0, 3.0])
g_d = -np.array([3.0, 0.0, 9.0]) / 2.0
x_star_d = np.array([1.0, 0.0, 1.0]) / 2.0
q_d = -2.5
r_d = -11.0 / 6.0

# 4) as 3) with the goldfarb variant, M = I: x = ( 3, 0, 4 ) / 5 has
# ||x|| = 1, and with lambda = 3, H + 3 I = diag( 4, 1, 6 ) and
# g = - ( 12, 0, 24 ) / 5, with q = - 132/25 + 57/50 = - 207/50
g_g = -np.array([12.0, 0.0, 24.0]) / 5.0
x_star_g = np.array([3.0, 0.0, 4.0]) / 5.0
q_g = -207.0 / 50.0

H_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

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

def solve(g, radius=None, weight=None, power=None, H=H, H_type='coordinate',
          f=0.0, H_data=None, **opts):
    # solve the trust-region problem if radius is given, and otherwise the
    # regularized one, returning x and inform. H_data = ( type, ne, val )
    # gives H directly, for the storage schemes without row and column
    # indices
    options = dps.initialize()
    options['print_level'] = 0
    # use the LAPACK solver sytr rather than the default, which needs the
    # environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    options['symmetric_linear_solver'] = 'sytr'
    options.update(opts)
    if H_data is None:
        H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, H_type)
    else:
        (H_type, H_ne, H_val), H_row, H_col, H_ptr = H_data, None, None, None
    dps.load(n, H_type, H_ne, H_row, H_col, H_ptr, options)
    if radius is not None:
        x = dps.solve_tr_problem(n, radius, f, g, H_ne, H_val)
    else:
        x = dps.solve_rq_problem(n, weight, power, f, g, H_ne, H_val)
    inform = dps.information()
    dps.terminate()
    return x, inform

def check_solution(x, inform, g, H, M, lam, f=0.0):
    # check ( H + lambda M ) x = - g for the known M, and the reported
    # ||x||_M and q(x)
    assert inform['multiplier'] == pytest.approx(lam, abs=1e-8)
    assert (H + lam * M) @ x + g == pytest.approx(np.zeros(n), abs=1e-8)
    assert np.linalg.eigvalsh(H + lam * M).min() >= -1e-8
    assert inform['x_norm'] == pytest.approx(np.sqrt(x @ M @ x), abs=1e-8)
    assert inform['obj'] == pytest.approx(f + g @ x + 0.5 * x @ H @ x,
                                          abs=1e-8)


@pytest.mark.parametrize('H_type', H_STORAGE)
@pytest.mark.parametrize('problem', ['tr', 'rq'])
def test_dps_hard_case(problem, H_type):
    # problem 1)
    if problem == 'tr':
        x, inform = solve(g, radius=1.0, H_type=H_type, f=f)
    else:
        x, inform = solve(g, weight=1.0, power=3.0, H_type=H_type, f=f)
        assert inform['obj_regularized'] == pytest.approx(r_hard, abs=1e-8)
    assert inform['status'] == 0
    assert inform['hard_case']
    assert inform['multiplier'] == pytest.approx(1.0, abs=1e-8)
    assert inform['x_norm'] == pytest.approx(1.0, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_hard, abs=1e-8)


# sytr factorizes H in 1) with a row interchange, and x is then recovered
# wrongly, so the reported q differs from q(x), also for g = ( 1, 2, 4 ),
# which is not a hard case
@pytest.mark.xfail(strict=True, reason='dps returns the wrong x when sytr '
                   'pivots (see https://github.com/ralna/GALAHAD/issues/632)')
@pytest.mark.parametrize('g_p', [g, np.array([1.0, 2.0, 4.0])],
                         ids=['hard', 'easy'])
@pytest.mark.parametrize('problem', ['tr', 'rq'])
def test_dps_pivoting(problem, g_p):
    if problem == 'tr':
        x, inform = solve(g_p, radius=1.0, f=f)
    else:
        x, inform = solve(g_p, weight=1.0, power=3.0, f=f)
    assert inform['status'] == 0
    assert inform['obj'] == pytest.approx(f + g_p @ x + 0.5 * x @ H @ x,
                                          abs=1e-8)


@pytest.mark.parametrize('H_type', H_STORAGE)
@pytest.mark.parametrize('radius, t', PD_TR, ids=PD_TR_IDS)
def test_dps_positive_definite_tr(radius, t, H_type):
    # problem 2), trust region
    x_star, q_star, lam = pd_values(t)
    x, inform = solve(g_pd, radius=radius, H=H_pd, H_type=H_type)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_star, abs=1e-8)
    check_solution(x, inform, g_pd, H_pd, H_pd, lam)


@pytest.mark.parametrize('H_type', H_STORAGE)
@pytest.mark.parametrize('power, weight, t', PD_RQ, ids=PD_RQ_IDS)
def test_dps_positive_definite_rq(power, weight, t, H_type):
    # problem 2), regularization
    x_star, q_star, lam = pd_values(t)
    x, inform = solve(g_pd, weight=weight, power=power, H=H_pd,
                      H_type=H_type)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_star, abs=1e-8)
    norm_x = t * norm_n
    assert lam == pytest.approx(weight * norm_x ** (power - 2.0), abs=1e-12)
    assert inform['obj_regularized'] \
        == pytest.approx(q_star + weight / power * norm_x ** power, abs=1e-8)
    check_solution(x, inform, g_pd, H_pd, H_pd, lam)


@pytest.mark.parametrize('H_type', H_STORAGE)
@pytest.mark.parametrize('problem', ['tr', 'rq'])
def test_dps_diagonal(problem, H_type):
    # problem 3)
    if problem == 'tr':
        x, inform = solve(g_d, radius=1.0, H=H_d, H_type=H_type)
    else:
        x, inform = solve(g_d, weight=2.0, power=3.0, H=H_d, H_type=H_type)
        assert inform['obj_regularized'] == pytest.approx(r_d, abs=1e-8)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_d, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_d, abs=1e-8)
    check_solution(x, inform, g_d, H_d, M_d, 2.0)


@pytest.mark.parametrize('H_type', H_STORAGE)
def test_dps_goldfarb(H_type):
    # problem 4)
    x, inform = solve(g_g, radius=1.0, H=H_d, H_type=H_type, goldfarb=True)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_g, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_g, abs=1e-8)
    check_solution(x, inform, g_g, H_d, np.eye(n), 3.0)


# 5) the goldfarb variant with H = diag( 1, 2, 3 ) (so M = I),
# g = ( 1, 1, 1 ) and radius 10: the minimizer x = - ( 1, 1/2, 1/3 ) of q
# lies inside the region, with lambda = 0 and q = - 11/12
@pytest.mark.xfail(strict=True, reason='dps (through trs_solve_diagonal) '
                   'returns a boundary point with a negative multiplier '
                   'instead of an interior solution (see '
                   'https://github.com/ralna/GALAHAD/pull/631)')
def test_dps_goldfarb_interior():
    H_i = np.diag([1.0, 2.0, 3.0])
    g_i = np.array([1.0, 1.0, 1.0])
    x, inform = solve(g_i, radius=10.0, H=H_i, goldfarb=True)
    assert x == pytest.approx(-np.array([1.0, 0.5, 1.0 / 3.0]), abs=1e-8)
    assert inform['obj'] == pytest.approx(-11.0 / 12.0, abs=1e-8)
    check_solution(x, inform, g_i, H_i, np.eye(n), 0.0)
    assert inform['status'] == 0


# 6) the singular H = diag( 1, 0, 3 ) with eigen_min = 1/2, so that
# M = diag( 1, 1/2, 3 ): x = ( 1/2, 1, sqrt( 3 ) / 6 ) has ||x||_M = 1, and
# with lambda = 2, H + 2 M = diag( 3, 1, 9 ) and
# g = - ( 3/2, 1, 3 sqrt( 3 ) / 2 ). Also with the second and third
# variables swapped, when the zero pivot comes last
SINGULAR_XFAIL = pytest.mark.xfail(
    strict=True, reason='dps treats the last pivots rather than the zero '
    'ones as zero when H is singular (see '
    'https://github.com/ralna/GALAHAD/issues/632)')
H_s = np.diag([1.0, 0.0, 3.0])
M_s = np.diag([1.0, 0.5, 3.0])
g_s = -np.array([1.5, 1.0, 1.5 * np.sqrt(3.0)])
x_star_s = np.array([0.5, 1.0, np.sqrt(3.0) / 6.0])

@pytest.mark.parametrize('order', [pytest.param([0, 1, 2],
                                                marks=SINGULAR_XFAIL),
                                   [0, 2, 1]],
                         ids=['zero-middle', 'zero-last'])
def test_dps_singular(order):
    P = np.array(order)
    Hp, Mp, gp = H_s[np.ix_(P, P)], M_s[np.ix_(P, P)], g_s[P]
    x, inform = solve(gp, radius=1.0, H=Hp, eigen_min=0.5)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_s[P], abs=1e-8)
    check_solution(x, inform, gp, Hp, Mp, 2.0)


# 7) H = s I, given by the storage schemes without indices, with
# g = ( 3, 0, 4 ) and radius 1. For s > 0, M = s I, and x = - t g with
# ||x||_M = 5 t sqrt( s ) = 1 and ( 1 + lambda ) s t = 1, so
# lambda = 5 / sqrt( s ) - 1; for s = 0 and eigen_min = 1, M = I,
# x = - g / 5 and lambda = 5. Also problem 3) in diagonal storage
TRIVIAL_XFAIL = pytest.mark.xfail(
    strict=True, reason='dps fails with status -40 for H in diagonal, '
    'scaled_identity, identity or zero storage (see '
    'https://github.com/ralna/GALAHAD/issues/632)')
g_t = np.array([3.0, 0.0, 4.0])
H_TRIVIAL = [('scaled_identity', 2.0, 1, np.array([2.0])),
             ('identity', 1.0, 0, None), ('zero', 0.0, 0, None)]

@TRIVIAL_XFAIL
@pytest.mark.parametrize('H_type, s, H_ne, H_val', H_TRIVIAL,
                         ids=[h[0] for h in H_TRIVIAL])
def test_dps_h_identity(H_type, s, H_ne, H_val):
    x, inform = solve(g_t, radius=1.0, H_data=(H_type, H_ne, H_val),
                      eigen_min=1.0)
    if s > 0.0:
        x_star, lam, M = - g_t / (5.0 * np.sqrt(s)), 5.0 / np.sqrt(s) - 1.0, \
            s * np.eye(n)
    else:
        x_star, lam, M = - g_t / 5.0, 5.0, np.eye(n)
    assert x == pytest.approx(x_star, abs=1e-8)
    check_solution(x, inform, g_t, s * np.eye(n), M, lam)
    assert inform['status'] == 0


@TRIVIAL_XFAIL
def test_dps_diagonal_storage():
    x, inform = solve(g_d, radius=1.0, H_data=('diagonal', n, np.diag(H_d)))
    assert x == pytest.approx(x_star_d, abs=1e-8)
    check_solution(x, inform, g_d, H_d, M_d, 2.0)
    assert inform['status'] == 0


@pytest.mark.parametrize('problem, a, b', [('tr', 0.0, None),
                                           ('tr', -1.0, None),
                                           ('rq', 0.0, 3.0),
                                           ('rq', 1.0, 1.5)],
                         ids=['radius=0', 'radius<0', 'sigma=0', 'p<2'])
def test_dps_bad_parameters(problem, a, b):
    # radius and sigma must be positive and p >= 2 (status -3)
    options = dps.initialize()
    options['print_level'] = 0
    H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, 'coordinate')
    try:
        dps.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, options)
        with pytest.raises(ValueError):
            if problem == 'tr':
                dps.solve_tr_problem(n, a, 0.0, g, H_ne, H_val)
            else:
                dps.solve_rq_problem(n, a, b, 0.0, g, H_ne, H_val)
        assert dps.information()['status'] == -3
    finally:
        dps.terminate()


def test_dps_resolve_tr():
    # problem 2): an interior solve, then a smaller radius, then g / 2
    options = dps.initialize()
    options['print_level'] = 0
    options['symmetric_linear_solver'] = 'sytr'
    H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H_pd, 'coordinate')
    try:
        dps.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, options)
        x = dps.solve_tr_problem(n, 2.0, 0.0, g_pd, H_ne, H_val)
        assert x == pytest.approx(x_n, abs=1e-8)
        t = 0.5 / norm_n
        x = dps.resolve_tr_problem(n, 0.5, 0.0, g_pd)
        inform = dps.information()
        assert inform['status'] == 0
        x_star, q_star, lam = pd_values(t)
        assert x == pytest.approx(x_star, abs=1e-8)
        assert inform['obj'] == pytest.approx(q_star, abs=1e-8)
        check_solution(x, inform, g_pd, H_pd, H_pd, lam)
        x = dps.resolve_tr_problem(n, 2.0, 0.0, g_pd / 2.0)
        inform = dps.information()
        assert inform['status'] == 0
        assert x == pytest.approx(x_n / 2.0, abs=1e-8)
        assert inform['obj'] == pytest.approx(-13.0 / 72.0, abs=1e-8)
        check_solution(x, inform, g_pd / 2.0, H_pd, H_pd, 0.0)
    finally:
        dps.terminate()


def test_dps_resolve_rq():
    # problem 2) with p = 3: sigma = 1, then sigma = 2
    options = dps.initialize()
    options['print_level'] = 0
    options['symmetric_linear_solver'] = 'sytr'
    H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H_pd, 'coordinate')
    try:
        dps.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, options)
        dps.solve_rq_problem(n, 1.0, 3.0, 0.0, g_pd, H_ne, H_val)
        x = dps.resolve_rq_problem(n, 2.0, 3.0, 0.0, g_pd)
        inform = dps.information()
        assert inform['status'] == 0
        lam = lambda_p3(2.0)
        x_star, q_star, _ = pd_values(1.0 / (1.0 + lam))
        assert x == pytest.approx(x_star, abs=1e-8)
        assert inform['obj'] == pytest.approx(q_star, abs=1e-8)
        norm_x = norm_n / (1.0 + lam)
        assert inform['obj_regularized'] \
            == pytest.approx(q_star + 2.0 / 3.0 * norm_x ** 3, abs=1e-8)
        check_solution(x, inform, g_pd, H_pd, H_pd, lam)
    finally:
        dps.terminate()
