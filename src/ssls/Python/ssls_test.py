from galahad import ssls
import numpy as np
import pytest


# ssls solves the block system
#   K ( x ; y ) = ( a ; b ),  K = ( H  A^T ; A  -C ),
# by factorizing K with sls. solve_system overwrites the right-hand side by
# the solution and returns it.

# 1) as in test_ssls.py: H = ( 1 0 0 ; 0 2 1 ; 0 1 3 ), A = ( 2 1 0 ; 0 1 1 )
# and C = 0: ( x, y ) = ( 1, 1, 1, 1, 1 ) for ( a, b ) = ( 3, 5, 5, 3, 2 )
H = np.array([[1.0, 0.0, 0.0],
              [0.0, 2.0, 1.0],
              [0.0, 1.0, 3.0]])
A = np.array([[2.0, 1.0, 0.0],
              [0.0, 1.0, 1.0]])
n, m = 3, 2
rhs_e = np.array([3.0, 5.0, 5.0, 3.0, 2.0])
sol_e = np.ones(n + m)

# 2) the solution z = ( x, y ) = ( 1, 2, 3, 1, -1 ), with the right-hand side
# K z formed for each H, A and C. K has n positive and m negative
# eigenvalues when H is positive definite
z_star = np.array([1.0, 2.0, 3.0, 1.0, -1.0])
C_s = np.array([[2.0, 1.0],
                [1.0, 3.0]])
ZERO_C = np.zeros((m, m))

def K(H=H, A=A, C=ZERO_C):
    return np.block([[H, A.T], [A, -C]])

def matrix_storage(M, M_type, symmetric):
    # return M_ne, M_row, M_col, M_ptr, M_val for M (its lower triangle if
    # symmetric) in the given storage scheme
    if M_type == 'dense':
        if symmetric:
            rows, cols = np.tril_indices(M.shape[0])
        else:
            rows, cols = np.indices(M.shape).reshape(2, -1)
        return len(rows), None, None, None, M[rows, cols]
    if M_type == 'diagonal':
        return M.shape[0], None, None, None, np.diag(M).copy()
    if M_type == 'scaled_identity':
        return 1, None, None, None, np.array([M[0, 0]])
    if M_type in ['identity', 'zero', 'none', 'absent']:
        return 0, None, None, None, None
    rows, cols = np.nonzero(np.tril(M) if symmetric else M)
    if M_type == 'coordinate':
        return len(rows), rows, cols, None, M[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=M.shape[0]))]
    return len(rows), None, cols, ptr, M[rows, cols]

def options():
    o = ssls.initialize()
    o['print_level'] = 0
    # use the LAPACK solver sytr rather than the default, which needs the
    # environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    o['symmetric_linear_solver'] = 'sytr'
    return o

@pytest.fixture
def terminate():
    yield
    ssls.terminate()

def factorize(H=H, A=A, C=ZERO_C, H_type='coordinate', A_type='coordinate',
              C_type='zero'):
    nn, mm = A.shape[1], A.shape[0]
    H_ne, H_row, H_col, H_ptr, H_val = matrix_storage(H, H_type, True)
    A_ne, A_row, A_col, A_ptr, A_val = matrix_storage(A, A_type, False)
    C_ne, C_row, C_col, C_ptr, C_val = matrix_storage(C, C_type, True)
    ssls.load(nn, mm, H_type, H_ne, H_row, H_col, H_ptr,
              A_type, A_ne, A_row, A_col, A_ptr,
              C_type, C_ne, C_row, C_col, C_ptr, options())
    ssls.factorize_matrix(H_ne, H_val, A_ne, A_val, C_ne, C_val)

def solve(rhs, **kw):
    # factorize and solve; return the solution, the right-hand side array
    # passed and inform
    factorize(**kw)
    A_kw = kw.get('A', A)
    rhs_c = rhs.copy()
    sol = ssls.solve_system(A_kw.shape[1], A_kw.shape[0], rhs_c)
    return sol, rhs_c, ssls.information()


@pytest.fixture
def solved(terminate):
    # problem 1)
    return solve(rhs_e)

def test_ssls_status(solved):
    sol, rhs_c, inform = solved
    assert inform['status'] == 0

def test_ssls_solution(solved):
    sol, rhs_c, inform = solved
    assert sol == pytest.approx(sol_e, abs=1e-8)
    assert K() @ sol == pytest.approx(rhs_e, abs=1e-8)

def test_ssls_in_place(solved):
    # solve_system overwrites the right-hand side by the solution, and
    # returns it
    sol, rhs_c, inform = solved
    assert sol is rhs_c

def test_ssls_inform(solved):
    sol, rhs_c, inform = solved
    assert inform['rank'] == n + m
    assert inform['rank_def'] is False
    assert inform['sls_inform']['negative_eigenvalues'] == m


# 2) storage schemes for H, A and C
STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

@pytest.mark.parametrize('C_type', STORAGE)
@pytest.mark.parametrize('A_type', STORAGE)
@pytest.mark.parametrize('H_type', STORAGE)
def test_ssls_storage(terminate, H_type, A_type, C_type):
    sol, rhs_c, inform = solve(K(C=C_s) @ z_star, C=C_s, H_type=H_type,
                               A_type=A_type, C_type=C_type)
    assert inform['status'] == 0
    assert sol == pytest.approx(z_star, abs=1e-8)

# H and C given by the schemes without indices: H = diag( 1, 2, 3 ), 2 I and
# I, and C = diag( 2, 3 ), 2 I, I and 0
H_TRIVIAL = [('diagonal', np.diag([1.0, 2.0, 3.0])),
             ('scaled_identity', 2.0 * np.eye(n)),
             ('identity', np.eye(n))]
C_TRIVIAL = [('diagonal', np.diag([2.0, 3.0])),
             ('scaled_identity', 2.0 * np.eye(m)),
             ('identity', np.eye(m)), ('zero', ZERO_C), ('none', ZERO_C)]

@pytest.mark.parametrize('H_type, H_t', H_TRIVIAL,
                         ids=[h[0] for h in H_TRIVIAL])
def test_ssls_h_trivial(terminate, H_type, H_t):
    sol, rhs_c, inform = solve(K(H=H_t, C=C_s) @ z_star, H=H_t, C=C_s,
                               H_type=H_type, C_type='coordinate')
    assert inform['status'] == 0
    assert sol == pytest.approx(z_star, abs=1e-8)

@pytest.mark.parametrize('C_type, C_t', C_TRIVIAL,
                         ids=[c[0] for c in C_TRIVIAL])
def test_ssls_c_trivial(terminate, C_type, C_t):
    sol, rhs_c, inform = solve(K(C=C_t) @ z_star, C=C_t, C_type=C_type)
    assert inform['status'] == 0
    assert sol == pytest.approx(z_star, abs=1e-8)


# 3) singular K: H = 0 (A has a null space, so K has rank n + m - 1 = 4),
# and a rank-deficient A = ( 1 1 0 ; 2 2 0 ) with C = 0 (rank 4). Consistent
# right-hand sides K z have (non-unique) solutions; inconsistent ones give
# status -11
ZERO_H = np.zeros((n, n))
A_d = np.array([[1.0, 1.0, 0.0],
                [2.0, 2.0, 0.0]])
SINGULAR = [({'H': ZERO_H, 'H_type': 'zero'}, K(H=ZERO_H)),
            ({'H': ZERO_H, 'H_type': 'none', 'C': C_s, 'C_type': 'coordinate'},
             K(H=ZERO_H, C=C_s)),
            ({'A': A_d}, K(A=A_d))]
SINGULAR_IDS = ['H=0', 'H=0-C', 'A-rank-deficient']

@pytest.mark.parametrize('kw, KM', SINGULAR, ids=SINGULAR_IDS)
def test_ssls_singular_consistent(terminate, kw, KM):
    rhs = KM @ z_star
    sol, rhs_c, inform = solve(rhs, **kw)
    assert inform['status'] == 0
    assert inform['rank'] == n + m - 1
    assert inform['rank_def'] is True
    assert KM @ sol == pytest.approx(rhs, abs=1e-8)

@pytest.mark.parametrize('kw, KM', SINGULAR, ids=SINGULAR_IDS)
def test_ssls_singular_inconsistent(terminate, kw, KM):
    # ( 1, 1, 1, 1, 1 ) is not in the range of any of these K
    factorize(**kw)
    with pytest.warns(RuntimeWarning):
        ssls.solve_system(n, m, np.ones(n + m))
    assert ssls.information()['status'] == -11


def test_ssls_refactorize(terminate):
    # problem 2) with C_s, after a factorization with C = 0 and the same
    # sparsity pattern
    H_ne, H_row, H_col, H_ptr, H_val = matrix_storage(H, 'coordinate', True)
    A_ne, A_row, A_col, A_ptr, A_val = matrix_storage(A, 'coordinate', False)
    C_ne, C_row, C_col, C_ptr, C_val = matrix_storage(C_s, 'coordinate', True)
    ssls.load(n, m, 'coordinate', H_ne, H_row, H_col, H_ptr,
              'coordinate', A_ne, A_row, A_col, A_ptr,
              'coordinate', C_ne, C_row, C_col, C_ptr, options())
    ssls.factorize_matrix(H_ne, H_val, A_ne, A_val, C_ne, np.zeros(C_ne))
    sol = ssls.solve_system(n, m, rhs_e.copy())
    assert sol == pytest.approx(sol_e, abs=1e-8)
    ssls.factorize_matrix(H_ne, H_val, A_ne, A_val, C_ne, C_val)
    rhs = K(C=C_s) @ z_star
    sol = ssls.solve_system(n, m, rhs.copy())
    assert ssls.information()['status'] == 0
    assert sol == pytest.approx(z_star, abs=1e-8)


# 4) no constraints (m = 0): H x = a, with x = ( 1, 2, 3 ) for
# a = ( 1, 7, 11 ). A may be given without entries, or as 'absent'
A_0 = np.zeros((0, n))
a_0 = H @ z_star[:n]

def test_ssls_no_constraints(terminate):
    sol, rhs_c, inform = solve(a_0, A=A_0, C=np.zeros((0, 0)))
    assert inform['status'] == 0
    assert sol == pytest.approx(z_star[:n], abs=1e-8)

@pytest.mark.xfail(strict=True, reason='ssls_import rejects the absent '
                   'storage scheme for A (see '
                   'https://github.com/ralna/GALAHAD/pull/657)')
def test_ssls_absent(terminate):
    sol, rhs_c, inform = solve(a_0, A=A_0, A_type='absent',
                               C=np.zeros((0, 0)))
    assert inform['status'] == 0
    assert sol == pytest.approx(z_star[:n], abs=1e-8)
