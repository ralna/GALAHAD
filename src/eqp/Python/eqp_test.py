from galahad import eqp
import numpy as np
import pytest


# test problems:
# 1) minimize f + g^T x + 1/2 x^T H x subject to A x + c = 0, where
#                             ( 1 0 0 )
# f = 1, g = ( 0, 2, 0 ), H = ( 0 2 1 ),  A = ( 2 1 0 ),  c = ( 3, 0 )
#                             ( 0 1 3 )       ( 0 1 1 )
# (test_eqp.py uses H = I, which is the identity case below)
# 2) the shifted least-distance problem
# minimize f + g^T x + 1/2 sum_j w_j^2 ( x_j - x^0_j )^2 subject to
# A x + c = 0, with f, g, A and c as above
n = 3
m = 2
f = 1.0
g = np.array([0.0, 2.0, 0.0])
H = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 3.0]])
A = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
c = np.array([3.0, 0.0])

# 1) H is positive definite, so the minimizer is the unique solution of the
# KKT conditions H x + g = A^T y and A x + c = 0, which gives
# x = ( -14, -11, 11 ) / 13, y = ( -7, 22 ) / 13 and f = 25 / 26
x_star = np.array([-14.0, -11.0, 11.0]) / 13.0
y_star = np.array([-7.0, 22.0]) / 13.0
f_star = 25.0 / 26.0

H_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']
A_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

def h_storage(H, H_type):
    # return H_ne, H_row, H_col, H_ptr, H_val for the lower triangle of H
    # in the given storage scheme
    rows, cols = np.tril_indices(n)
    if H_type == 'dense':
        return len(rows), None, None, None, H[rows, cols]
    nz = H[rows, cols] != 0.0
    rows, cols = rows[nz], cols[nz]
    if H_type == 'coordinate':
        return len(rows), rows, cols, None, H[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))]
    return len(rows), None, cols, ptr, H[rows, cols]

def a_storage(A, A_type):
    # return A_ne, A_row, A_col, A_ptr, A_val for A in the given storage
    # scheme
    if A_type == 'dense':
        return A.size, None, None, None, A.ravel()
    rows, cols = np.nonzero(A)
    if A_type == 'coordinate':
        return len(rows), rows, cols, None, A[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=m))]
    return len(rows), None, cols, ptr, A[rows, cols]

def initialize():
    options = eqp.initialize()
    options['print_level'] = 0
    # use the LAPACK solver sytr rather than the defaults, which need the
    # environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    options['fdc_options']['use_sls'] = True
    options['fdc_options']['symmetric_linear_solver'] = 'sytr'
    options['sbls_options']['symmetric_linear_solver'] = 'sytr'
    options['sbls_options']['definite_linear_solver'] = 'sytr'
    return options

def solve_qp(H_type, H_ne, H_row, H_col, H_ptr, H_val, A_type, g):
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, A_type)
    options = initialize()
    eqp.load(n, m, H_type, H_ne, H_row, H_col, H_ptr,
             A_type, A_ne, A_row, A_col, A_ptr, options)
    x, y = eqp.solve_qp(n, m, f, g, H_ne, H_val, A_ne, A_val, c,
                        np.zeros(n), np.zeros(m))
    inform = eqp.information()
    eqp.terminate()
    return x, y, inform

def check_optimality(x, y, grad):
    # x is feasible, A x + c = 0, and the first-order condition
    # grad = A^T y holds
    assert A @ x + c == pytest.approx(np.zeros(m), abs=1e-8)
    assert grad == pytest.approx(A.T @ y, abs=1e-8)

@pytest.fixture(params=[(H_type, A_type) for H_type in H_STORAGE
                        for A_type in A_STORAGE],
                ids=lambda p: f'H={p[0]},A={p[1]}')
def solution(request):
    H_type, A_type = request.param
    return solve_qp(H_type, *h_storage(H, H_type), A_type, g)


def test_eqp_status(solution):
    x, y, inform = solution
    assert inform['status'] == 0


def test_eqp_solution(solution):
    x, y, inform = solution
    assert inform['obj'] == pytest.approx(f + g @ x + 0.5 * x @ H @ x,
                                          abs=1e-8)
    check_optimality(x, y, H @ x + g)
    assert x == pytest.approx(x_star, abs=1e-8)
    assert y == pytest.approx(y_star, abs=1e-8)
    assert inform['obj'] == pytest.approx(f_star, abs=1e-8)


# the diagonal, scaled_identity and identity schemes cannot store the H
# above, so solve the problem with H = diag(d) instead, given H_type, H_ne
# and H_val for each case. There are no values to give for the identity,
# so H_val may be empty or None. Solving diag(d) x + g = A^T y and
# A x + c = 0 gives x, y and f; the identity case is problem 1 of
# test_eqp.py
SEPARABLE = {
    'diagonal': ('diagonal', np.array([1.0, 2.0, 3.0]), 3,
                 np.array([1.0, 2.0, 3.0]),
                 np.array([-26.0, -11.0, 11.0]) / 21.0,
                 np.array([-13.0 / 21.0, 11.0 / 7.0]), 59.0 / 42.0),
    'scaled_identity': ('scaled_identity', np.array([2.0, 2.0, 2.0]), 1,
                        np.array([2.0]), np.array([-10.0, -7.0, 7.0]) / 9.0,
                        np.array([-10.0, 14.0]) / 9.0, 17.0 / 9.0),
    'identity': ('identity', np.array([1.0, 1.0, 1.0]), 0, np.zeros(0),
                 np.array([-8.0, -11.0, 11.0]) / 9.0,
                 np.array([-4.0, 11.0]) / 9.0, 4.0 / 9.0),
    'identity-None': ('identity', np.array([1.0, 1.0, 1.0]), 0, None,
                      np.array([-8.0, -11.0, 11.0]) / 9.0,
                      np.array([-4.0, 11.0]) / 9.0, 4.0 / 9.0),
}

@pytest.mark.parametrize('case', SEPARABLE)
def test_eqp_separable(case):
    H_type, d, H_ne, H_val, x_star_s, y_star_s, f_star_s = SEPARABLE[case]
    x, y, inform \
        = solve_qp(H_type, H_ne, None, None, None, H_val, 'coordinate', g)
    assert inform['status'] == 0
    assert inform['obj'] == pytest.approx(f + g @ x + 0.5 * d @ x**2,
                                          abs=1e-8)
    check_optimality(x, y, d * x + g)
    assert x == pytest.approx(x_star_s, abs=1e-8)
    assert y == pytest.approx(y_star_s, abs=1e-8)
    assert inform['obj'] == pytest.approx(f_star_s, abs=1e-8)


# with H = 0 (the zero and none schemes), g^T x is unbounded below on the
# feasible set unless g lies in the range of A^T. So use g = A^T ( 1, 1 )
# = ( 2, 2, 1 ), for which g^T x = ( 1, 1 ) A x = -( 1, 1 ) c = -3 for every
# feasible x. Every feasible x is then a minimizer, with y = ( 1, 1 ) and
# f = 1 - 3 = -2
ZERO = {
    'zero': ('zero', np.zeros(0)),
    'zero-None': ('zero', None),
    'none': ('none', np.zeros(0)),
    'none-None': ('none', None),
}

@pytest.mark.parametrize('case', ZERO)
def test_eqp_zero(case):
    H_type, H_val = ZERO[case]
    g_0 = A.T @ np.ones(m)
    x, y, inform \
        = solve_qp(H_type, 0, None, None, None, H_val, 'coordinate', g_0)
    assert inform['status'] == 0
    check_optimality(x, y, g_0)
    assert y == pytest.approx(np.ones(m), abs=1e-8)
    assert inform['obj'] == pytest.approx(f + g_0 @ x, abs=1e-8)
    assert inform['obj'] == pytest.approx(-2.0, abs=1e-8)


# 2) with W = diag(w), the problem is problem 1 with H = W^2,
# g - W^2 x^0 in place of g and a different constant, so solving the KKT
# conditions W^2 ( x - x^0 ) + g = A^T y and A x + c = 0 gives x, y and f.
# test_eqp.py uses w = ( 1, 1, 1 ) and x^0 = 0, which is the identity case
# above, so also try w = ( 1, 2, 3 ) and x^0 = ( 1, 1, 1 )
SLD = {
    'w=1,x0=0': (np.ones(n), np.zeros(n), np.array([-8.0, -11.0, 11.0]) / 9.0,
                 np.array([-4.0, 11.0]) / 9.0, 4.0 / 9.0),
    'w=123,x0=1': (np.array([1.0, 2.0, 3.0]), np.ones(n),
                   np.array([-63.0, -33.0, 33.0]) / 53.0,
                   np.array([-58.0, -180.0]) / 53.0, 427.0 / 53.0),
}

@pytest.mark.parametrize('case', SLD)
def test_eqp_sldqp(case):
    w, x0, x_star_sld, y_star_sld, f_star_sld = SLD[case]
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, 'coordinate')
    options = initialize()
    eqp.load(n, m, 'shifted_least_distance', 0, None, None, None,
             'coordinate', A_ne, A_row, A_col, A_ptr, options)
    x, y = eqp.solve_sldqp(n, m, f, g, w, x0, A_ne, A_val, c,
                           np.zeros(n), np.zeros(m))
    inform = eqp.information()
    eqp.terminate()
    assert inform['status'] == 0
    assert inform['obj'] == pytest.approx(
        f + g @ x + 0.5 * np.sum(w**2 * (x - x0)**2), abs=1e-8)
    check_optimality(x, y, w**2 * (x - x0) + g)
    assert x == pytest.approx(x_star_sld, abs=1e-8)
    assert y == pytest.approx(y_star_sld, abs=1e-8)
    assert inform['obj'] == pytest.approx(f_star_sld, abs=1e-8)
