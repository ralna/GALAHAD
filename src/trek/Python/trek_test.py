from galahad import trek
import numpy as np
import pytest


# trek finds the global minimizer of q(x) = f + c^T x + 1/2 x^T H x within
# the trust region ||x||_S <= radius, where ||x||_S = sqrt( x^T S x ), by an
# extended-Krylov-subspace method. x solves the problem if, for some
# lambda >= 0, ( H + lambda S ) x = - c, lambda ( ||x||_S - radius ) = 0,
# ||x||_S <= radius and H + lambda S is positive semi-definite. f is
# options['f'] (zero by default); inform reports lambda ('multiplier'), q
# ('obj') and ||x||_S, and suggests a smaller radius for a re-solve
# ('next_radius', options['reduction'] times the smaller of the radius and
# ||x||_S).
#
# test problems, as in test_trek.py, with
#     ( 1 0 4 )
# H = ( 0 2 0 ), c = ( 0, 2, 0 ) and S = diag( 1, 2, 1 ):
#     ( 4 0 3 )
# 1) S = I, radius 1; 2) the same with radius 1/2; 3) S, radius 1.
# H has eigenvalues 2 and 2 +- sqrt( 17 ), the smallest with eigenvector
# u = ( 4, 0, 1 - sqrt( 17 ) ), which is orthogonal to c. These are all the
# hard case: lambda = sqrt( 17 ) - 2, minus the smallest eigenvalue of H
# (and of H with respect to S, as S is the identity on the span of e_0 and
# e_2), and ( H + lambda S ) x_1 = - c gives x_1 = - 2 / ( 2 + lambda s_1 ),
# with ||x_1 e_1||_S < radius. The solutions x = x_1 e_1 + tau u / ||u||
# (with tau chosen so that ||x||_S = radius, of either sign) are not unique,
# but lambda and q(x) are: as H e_1 = 2 e_1 and H u = - lambda u,
#   q = 2 x_1 + x_1^2 - 1/2 lambda tau^2, tau^2 = radius^2 - s_1 x_1^2
# where s_1 = 1 or 2. The Krylov space generated from c is just the span of
# e_1, so trek only finds these solutions if c is perturbed
# (options['perturb_c'])
n = 3
H = np.array([[1.0, 0.0, 4.0], [0.0, 2.0, 0.0], [4.0, 0.0, 3.0]])
S = np.diag([1.0, 2.0, 1.0])
c = np.array([0.0, 2.0, 0.0])
lambda_hard = np.sqrt(17.0) - 2.0

def hard_case_q(radius, s_1):
    x_1 = - 2.0 / (2.0 + lambda_hard * s_1)
    tau2 = radius ** 2 - s_1 * x_1 ** 2
    return 2.0 * x_1 + x_1 ** 2 - 0.5 * lambda_hard * tau2

# well-posed variants, built from a chosen x and lambda with H + lambda S
# positive definite by setting c = - ( H + lambda S ) x:
# 4) S = I, x = ( 2, -1, -2 ) / 3 with ||x|| = 1 and lambda = 3, so
# c = ( 0, 5, 4 ) / 3 and q = - 13/9 - 7/9 = - 20/9
c_b = np.array([0.0, 5.0, 4.0]) / 3.0
x_star_b = np.array([2.0, -1.0, -2.0]) / 3.0
q_b = -20.0 / 9.0
# 5) S as above, x = ( 1, 1, 1 ) / 2 with ||x||_S = 1 and lambda = 3, so
# c = - ( 4, 4, 5 ) and q = - 13/2 + 7/4 = - 19/4
c_s = -np.array([4.0, 4.0, 5.0])
x_star_s = np.array([0.5, 0.5, 0.5])
q_s = -19.0 / 4.0
# 6) the positive definite H + 3 I, with c and x as in 4), radius 2: the
# unconstrained minimizer x = ( 2, -1, -2 ) / 3 lies inside the region, so
# lambda = 0 and q = 1/2 c^T x = - 13/18
H_pd = H + 3.0 * np.eye(n)
q_i = -13.0 / 18.0

# 7) general radii, for which lambda solves the secular equation
# ||x(lambda)|| = radius, x(lambda) = - ( H + lambda I )^-1 c, for H and c
# as in 4), and for the even-dimensional H_4 = ( H 0 ; 0 5 ) and
# c_4 = ( c, 1 ) of 4); the reference is found by bisection
H_4 = np.zeros((4, 4))
H_4[:3, :3] = H
H_4[3, 3] = 5.0
c_4 = np.r_[c_b, 1.0]

def secular_solution(H, c, radius):
    # bisect for the root lambda > - lambda_min(H) of ||x(lambda)|| = radius
    n = len(c)
    lo, hi = max(0.0, -np.linalg.eigvalsh(H).min()), 1e6
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        x = np.linalg.solve(H + lam * np.eye(n), -c)
        lo, hi = (lam, hi) if np.linalg.norm(x) > radius else (lo, lam)
    return x, lam

H_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']
S_STORAGE = ['coordinate', 'sparse_by_rows', 'dense', 'diagonal']

ISSUE = 'https://github.com/ralna/GALAHAD/issues/644'

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

def options(**opts):
    o = trek.initialize()
    o['print_level'] = 0
    # use the LAPACK solvers sytr and potr rather than the defaults, which
    # need the environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    o['linear_solver'] = 'sytr'
    o['linear_solver_for_s'] = 'sytr'
    o['trs_options']['symmetric_linear_solver'] = 'sytr'
    o['trs_options']['definite_linear_solver'] = 'potr'
    # the default stopping tolerance only gives residuals of around 1e-8
    o['stop_residual'] = 1e-10
    o.update(opts)
    return o

def solve(c, radius, H=H, H_type='coordinate', S=None, S_type='coordinate',
          H_data=None, S_data=None, **opts):
    # solve the problem, returning x and inform. H_data or S_data =
    # ( type, ne, val ) give H or S directly, for the storage schemes
    # without indices
    if H_data is None:
        H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, H_type)
    else:
        (H_type, H_ne, H_val), H_row, H_col, H_ptr = H_data, None, None, None
    trek.load(n, H_type, H_ne, H_row, H_col, H_ptr, options(**opts))
    args = {}
    if S_data is not None:
        S_type, S_ne, S_val = S_data
        trek.load_s(n, S_type, S_ne, None, None, None)
        args = dict(S_ne=S_ne, S_val=S_val)
    elif S is not None:
        S_ne, S_row, S_col, S_ptr, S_val = sym_storage(S, S_type)
        trek.load_s(n, S_type, S_ne, S_row, S_col, S_ptr)
        args = dict(S_ne=S_ne, S_val=S_val)
    x = trek.solve_problem(n, H_ne, H_val, c, radius, **args)
    return x, trek.information()

def check_solution(x, inform, c, radius, H=H, S=None, f=0.0):
    # check the conditions above for the reported multiplier, and the
    # reported q
    S = np.eye(n) if S is None else S
    lam = inform['multiplier']
    norm_x = np.sqrt(x @ S @ x)
    assert lam >= 0.0
    assert norm_x <= radius + 1e-8
    assert lam * (norm_x - radius) == pytest.approx(0.0, abs=1e-8)
    assert (H + lam * S) @ x + c == pytest.approx(np.zeros(n), abs=1e-8)
    assert np.linalg.eigvalsh(H + lam * S).min() >= -1e-8
    assert inform['obj'] == pytest.approx(f + c @ x + 0.5 * x @ H @ x,
                                          abs=1e-8)


@pytest.fixture
def terminate():
    yield
    trek.terminate()


# for radius 1, trek only reaches the solution after many more vectors than
# the 2n that span the extended Krylov space, so the outcome depends on
# rounding: it succeeds on macOS, but runs out of iterations on Linux and
# Windows, hence a non-strict xfail
@pytest.mark.parametrize('problem', [
    pytest.param((1.0, False), marks=pytest.mark.xfail(
        strict=False, reason='trek carries on past the exhausted extended '
        f'Krylov space, with a platform-dependent outcome (see {ISSUE})')),
    (0.5, False), (1.0, True)],
    ids=['radius=1', 'radius=1/2', 'S'])
def test_trek_hard_case(terminate, problem):
    # problems 1) to 3), with c perturbed (with the default stop_residual,
    # trek stops at a point that is not the global minimizer for problem 2))
    radius, use_s = problem
    Sx = S if use_s else None
    x, inform = solve(c, radius, S=Sx, perturb_c=True)
    assert inform['status'] == 0
    assert inform['multiplier'] == pytest.approx(lambda_hard, abs=1e-6)
    assert inform['obj'] == pytest.approx(
        hard_case_q(radius, 2.0 if use_s else 1.0), abs=1e-6)
    assert inform['x_norm'] == pytest.approx(radius, abs=1e-8)


# without the perturbation, trek stops at a first-order point in the span of
# e_1 that is not the global minimizer, or (with S) fails with NaN values
@pytest.mark.filterwarnings('ignore:too many iterations')
@pytest.mark.xfail(strict=True, reason='trek does not detect the hard case '
                   f'unless c is perturbed (see {ISSUE})')
@pytest.mark.parametrize('use_s', [False, True], ids=['I', 'S'])
def test_trek_hard_case_unperturbed(terminate, use_s):
    Sx = S if use_s else None
    x, inform = solve(c, 1.0, S=Sx)
    assert inform['status'] == 0
    assert inform['obj'] == pytest.approx(
        hard_case_q(1.0, 2.0 if use_s else 1.0), abs=1e-6)


@pytest.mark.parametrize('H_type', H_STORAGE)
def test_trek_boundary(terminate, H_type):
    # problem 4)
    x, inform = solve(c_b, 1.0, H_type=H_type)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_b, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(3.0, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_b, abs=1e-8)
    assert inform['x_norm'] == pytest.approx(1.0, abs=1e-8)
    check_solution(x, inform, c_b, 1.0)


@pytest.mark.parametrize('S_type', S_STORAGE)
def test_trek_scaled(terminate, S_type):
    # problem 5)
    x, inform = solve(c_s, 1.0, S=S, S_type=S_type)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_s, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(3.0, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_s, abs=1e-8)
    assert inform['x_norm'] == pytest.approx(1.0, abs=1e-8)
    check_solution(x, inform, c_s, 1.0, S=S)


@pytest.mark.parametrize('use_s', [False, True], ids=['I', 'S'])
def test_trek_interior(terminate, use_s):
    # problem 6), with ||x|| = 1 and ||x||_S = sqrt( 10 ) / 3 < radius = 2
    radius = 2.0
    Sx = S if use_s else None
    x, inform = solve(c_b, radius, H=H_pd, S=Sx)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_b, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(0.0, abs=1e-8)
    assert inform['obj'] == pytest.approx(q_i, abs=1e-8)
    check_solution(x, inform, c_b, radius, H=H_pd, S=Sx)


def test_trek_interior_x_norm(terminate):
    # problem 6): the reported ||x||
    x, inform = solve(c_b, 2.0, H=H_pd)
    assert inform['x_norm'] == pytest.approx(np.linalg.norm(x_star_b),
                                             abs=1e-8)


# for odd n, trek carries on past an exhausted extended Krylov space, and
# returns an inaccurate solution with status 0
@pytest.mark.parametrize('problem', [
    pytest.param((H, c_b), marks=pytest.mark.xfail(
        strict=True, reason='trek returns an inaccurate solution when n is '
        f'odd (see {ISSUE})')),
    (H_4, c_4)],
    ids=['n=3', 'n=4'])
@pytest.mark.parametrize('radius', [0.5, 1.5])
def test_trek_secular(terminate, problem, radius):
    # problem 7)
    H, c = problem
    n = len(c)
    x_ref, lam_ref = secular_solution(H, c, radius)
    H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, 'coordinate')
    trek.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, options())
    x = trek.solve_problem(n, H_ne, H_val, c, radius)
    inform = trek.information()
    assert inform['status'] == 0
    assert x == pytest.approx(x_ref, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(lam_ref, abs=1e-8)


def test_trek_objective_constant(terminate):
    # problem 4) with f = 1/2
    x, inform = solve(c_b, 1.0, f=0.5)
    assert inform['obj'] == pytest.approx(q_b + 0.5, abs=1e-8)
    check_solution(x, inform, c_b, 1.0, f=0.5)


def test_trek_next_radius(terminate):
    # the suggested radius is reduction times the smaller of the radius and
    # ||x||_S
    x, inform = solve(c_b, 1.0, reduction=0.25)
    assert inform['next_radius'] == pytest.approx(0.25, abs=1e-8)


def test_trek_resolve(terminate):
    # problem 4) with radius 1, re-solved (new_radius) after a solve with
    # radius 2
    H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, 'coordinate')
    o = options()
    trek.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, o)
    x = trek.solve_problem(n, H_ne, H_val, c_b, 2.0)
    assert trek.information()['status'] == 0
    o['new_radius'] = True
    trek.reset_options(o)
    x = trek.solve_problem(n, H_ne, H_val, c_b, 1.0)
    inform = trek.information()
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_b, abs=1e-8)
    check_solution(x, inform, c_b, 1.0)


def test_trek_resolve_first(terminate):
    # a re-solve (new_radius) must follow a solve (status -31, which is
    # reported only through inform)
    H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, 'coordinate')
    trek.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr,
              options(new_radius=True))
    trek.solve_problem(n, H_ne, H_val, c_b, 1.0)
    assert trek.information()['status'] == -31


def test_trek_missing_s_val(terminate):
    # S_val must be given once S has been loaded (status -79, which is
    # reported only through inform)
    H_ne, H_row, H_col, H_ptr, H_val = sym_storage(H, 'coordinate')
    S_ne, S_row, S_col, S_ptr, S_val = sym_storage(S, 'coordinate')
    trek.load(n, 'coordinate', H_ne, H_row, H_col, H_ptr, options())
    trek.load_s(n, 'coordinate', S_ne, S_row, S_col, S_ptr)
    trek.solve_problem(n, H_ne, H_val, c_s, 1.0)
    assert trek.information()['status'] == -79


@pytest.mark.parametrize('radius', [0.0, -1.0], ids=['radius=0', 'radius<0'])
def test_trek_radius_not_positive(terminate, radius):
    # the radius must be positive (status -3)
    with pytest.raises(ValueError):
        solve(c_b, radius)
    assert trek.information()['status'] == -3


# 8) H = s I given by the storage schemes without indices, S = I and
# c = ( 3, 0, 4 ), with ||c|| = 5 > s radius for s <= 2: x = - c / 5 lies on
# the boundary, lambda = 5 - s and q = - 5 + s / 2
c_t = np.array([3.0, 0.0, 4.0])
H_IDENTITY = [('scaled_identity', 2.0, 1, np.array([2.0])),
              ('identity', 1.0, 0, None), ('zero', 0.0, 0, None),
              ('none', 0.0, 0, None)]

@pytest.mark.parametrize('H_type, s, H_ne, H_val', H_IDENTITY,
                         ids=[h[0] for h in H_IDENTITY])
def test_trek_h_identity(terminate, H_type, s, H_ne, H_val):
    x, inform = solve(c_t, 1.0, H_data=(H_type, H_ne, H_val))
    assert inform['status'] == 0
    assert x == pytest.approx(-c_t / 5.0, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(5.0 - s, abs=1e-8)
    assert inform['obj'] == pytest.approx(-5.0 + s / 2.0, abs=1e-8)


# 9) S = 4 I, so ||x||_S = 2 ||x||, with problem 4) scaled:
# x = ( 2, -1, -2 ) / 6 has ||x||_S = 1, and with lambda = 3 / 4, H + 3 I is
# again positive definite, so c = - ( H + 3 I ) x = ( 0, 5, 4 ) / 6
@pytest.mark.parametrize('S_data', [
    ('diagonal', 3, np.array([4.0, 4.0, 4.0])),
    ('scaled_identity', 1, np.array([4.0]))],
    ids=['diagonal', 'scaled_identity'])
def test_trek_scaled_identity(terminate, S_data):
    x, inform = solve(c_b / 2.0, 1.0, S_data=S_data)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_b / 2.0, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(0.75, abs=1e-8)
    check_solution(x, inform, c_b / 2.0, 1.0, S=4.0 * np.eye(n))


# check that S_val may be None for an identity S
@pytest.mark.parametrize('S_val', [np.zeros(0), None], ids=['empty', 'None'])
def test_trek_s_identity(terminate, S_val):
    # problem 4) with S = I given by the identity storage scheme
    x, inform = solve(c_b, 1.0, S_data=('identity', 0, S_val))
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_b, abs=1e-8)
    check_solution(x, inform, c_b, 1.0)
