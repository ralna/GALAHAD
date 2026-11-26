from galahad import sbls
from itertools import combinations
import numpy as np
import pytest


# sbls solves the block system
#   K_G ( x ; y ) = ( a ; b ),  K_G = ( G  A^T ; A  -C ),
# where G is an approximation to H chosen by options['preconditioner']:
# 1: G = I, 2: G = H, 3: G = diag( max( H_ii, min_diagonal ) ),
# 4: G = band( H ), 5: G = diag( D ), and 11, 12, -1, -2 partition
# A = ( A_1  A_2 ) for a nonsingular "basis" A_1 (chosen by sbls) and set
# G_11 = 0, G_21 = 0 (11, -1, -2) or H_21 (12), and G_22 = H_22 (11, 12,
# -2) or I (-1). The implicit preconditioners -1 and -2 are for C = 0.
# solve_system overwrites the right-hand side by the solution and returns it.

# 1) as in test_sbls.py: H = ( 1 0 0 ; 0 2 1 ; 0 1 3 ), A = ( 2 1 0 ; 0 1 1 )
# and C = 0, with G = H: ( x, y ) = ( 1, 1, 1, 1, 1 ) for
# ( a, b ) = ( 3, 5, 5, 3, 2 )
H = np.array([[1.0, 0.0, 0.0],
              [0.0, 2.0, 1.0],
              [0.0, 1.0, 3.0]])
A = np.array([[2.0, 1.0, 0.0],
              [0.0, 1.0, 1.0]])
n, m = 3, 2
rhs_e = np.array([3.0, 5.0, 5.0, 3.0, 2.0])
sol_e = np.ones(n + m)

# 2) the solution z = ( x, y ) = ( 1, 2, 3, 1, -1 ), with the right-hand side
# K_G z formed for each G and C
z_star = np.array([1.0, 2.0, 3.0, 1.0, -1.0])
C_s = np.array([[2.0, 1.0],
                [1.0, 3.0]])
D = np.array([1.0, 2.0, 3.0])

def K(G, C):
    return np.block([[G, A.T], [A, -C]])

def explicit_G(preconditioner):
    # G for the explicit preconditioners that do not depend on a basis
    # (band( H ) = H, as the default semi_bandwidth is 5)
    return {1: np.eye(n), 2: H, 3: np.diag(np.diag(H)), 4: H,
            5: np.diag(D)}[preconditioner]

def basis_G(preconditioner, basis):
    # G for the preconditioners defined by a basis A_1 = A[:, basis]
    rest = [j for j in range(n) if j not in basis]
    G = np.zeros((n, n))
    G[np.ix_(rest, rest)] = (np.eye(len(rest)) if preconditioner == -1
                             else H[np.ix_(rest, rest)])
    if preconditioner == 12:
        G[np.ix_(rest, basis)] = H[np.ix_(rest, basis)]
        G[np.ix_(basis, rest)] = H[np.ix_(basis, rest)]
    return G

BASES = [list(b) for b in combinations(range(n), m)
         if abs(np.linalg.det(A[:, list(b)])) > 1e-12]

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

def options(preconditioner=2, factorization=2, perturb=True):
    o = sbls.initialize()
    o['print_level'] = 0
    # use the LAPACK solvers rather than the defaults, which need the
    # environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    o['symmetric_linear_solver'] = 'sytr'
    o['definite_linear_solver'] = 'potr'
    o['unsymmetric_linear_solver'] = 'getr'
    o['preconditioner'] = preconditioner
    o['factorization'] = factorization
    o['perturb_to_make_definite'] = perturb
    return o

@pytest.fixture
def terminate():
    yield
    sbls.terminate()

def factorize(H=H, C=None, preconditioner=2, factorization=2,
              H_type='coordinate', A_type='coordinate', C_type=None,
              perturb=True, **kw):
    # load and factorize. C = None means C = 0 ('zero')
    if C is None:
        C, C_type = np.zeros((m, m)), C_type or 'zero'
    else:
        C_type = C_type or 'coordinate'
    H_ne, H_row, H_col, H_ptr, H_val = matrix_storage(H, H_type, True)
    A_ne, A_row, A_col, A_ptr, A_val = matrix_storage(A, A_type, False)
    C_ne, C_row, C_col, C_ptr, C_val = matrix_storage(C, C_type, True)
    sbls.load(n, m, H_type, H_ne, H_row, H_col, H_ptr,
              A_type, A_ne, A_row, A_col, A_ptr,
              C_type, C_ne, C_row, C_col, C_ptr,
              options(preconditioner, factorization, perturb))
    sbls.factorize_matrix(n, m, H_ne, H_val, A_ne, A_val, C_ne, C_val, **kw)

def solve(rhs, **kw):
    # load, factorize and solve; return the solution, the right-hand side
    # array passed and inform
    factorize(**kw)
    rhs_c = rhs.copy()
    sol = sbls.solve_system(n, m, rhs_c)
    return sol, rhs_c, sbls.information()


@pytest.fixture
def solved(terminate):
    # problem 1)
    return solve(rhs_e)

def test_sbls_status(solved):
    sol, rhs_c, inform = solved
    assert inform['status'] == 0
    assert inform['preconditioner'] == 2
    assert inform['factorization'] == 2

def test_sbls_solution(solved):
    sol, rhs_c, inform = solved
    assert sol == pytest.approx(sol_e, abs=1e-8)
    assert K(H, np.zeros((m, m))) @ sol == pytest.approx(rhs_e, abs=1e-8)

def test_sbls_in_place(solved):
    # solve_system overwrites the right-hand side by the solution, and
    # returns it
    sol, rhs_c, inform = solved
    assert sol is rhs_c

def test_sbls_inform(solved):
    sol, rhs_c, inform = solved
    assert inform['rank_def'] is False
    assert inform['rank'] == m


# 2) explicit preconditioners, with the Schur-complement (1) and
# augmented-system (2) factorizations, and C = 0 or C_s
@pytest.mark.parametrize('C', [None, C_s], ids=['C=0', 'C'])
@pytest.mark.parametrize('factorization', [1, 2])
@pytest.mark.parametrize('preconditioner', [1, 2, 3, 4, 5])
def test_sbls_explicit(terminate, preconditioner, factorization, C):
    G = explicit_G(preconditioner)
    Cm = np.zeros((m, m)) if C is None else C
    rhs = K(G, Cm) @ z_star
    kw = {'D': D} if preconditioner == 5 else {}
    sol, rhs_c, inform = solve(rhs, C=C, preconditioner=preconditioner,
                               factorization=factorization, **kw)
    assert inform['status'] == 0
    assert inform['preconditioner'] == preconditioner
    assert sol == pytest.approx(z_star, abs=1e-8)

@pytest.mark.parametrize('preconditioner', [1, 3, 5])
def test_sbls_schur_complement(terminate, preconditioner):
    # factorization 4 (Schur complement or failure) with a diagonal G
    G = explicit_G(preconditioner)
    rhs = K(G, C_s) @ z_star
    kw = {'D': D} if preconditioner == 5 else {}
    sol, rhs_c, inform = solve(rhs, C=C_s, preconditioner=preconditioner,
                               factorization=4, **kw)
    assert inform['status'] == 0
    assert inform['factorization'] == 1
    assert sol == pytest.approx(z_star, abs=1e-8)

@pytest.mark.parametrize('D_arg', ['omitted', None], ids=['omitted', 'None'])
def test_sbls_preconditioner_5_without_d(terminate, D_arg):
    # without D, preconditioner 5 becomes preconditioner 1 (G = I)
    rhs = K(np.eye(n), np.zeros((m, m))) @ z_star
    kw = {} if D_arg == 'omitted' else {'D': None}
    sol, rhs_c, inform = solve(rhs, preconditioner=5, **kw)
    assert inform['status'] == 0
    assert inform['preconditioner'] == 1
    assert sol == pytest.approx(z_star, abs=1e-8)


# 3) preconditioners defined by a basis A_1 of A: the solution must solve
# K_G z = rhs for the G of one of the possible bases (for which K_G is
# nonsingular). The implicit preconditioners are for C = 0
def basis_solutions(preconditioner, C, rhs):
    solutions = []
    for b in BASES:
        KG = K(basis_G(preconditioner, b), C)
        if np.linalg.cond(KG) < 1e12:
            solutions.append(np.linalg.solve(KG, rhs))
    return solutions

@pytest.mark.parametrize('preconditioner, C',
                         [(11, None), (12, None), (-1, None), (-2, None),
                          (11, C_s)],
                         ids=['11-C=0', '12-C=0', '-1-C=0', '-2-C=0', '11-C'])
def test_sbls_basis(terminate, preconditioner, C):
    Cm = np.zeros((m, m)) if C is None else C
    sol, rhs_c, inform = solve(rhs_e, C=C, preconditioner=preconditioner)
    assert inform['status'] == 0
    assert inform['preconditioner'] == preconditioner
    assert inform['perturbed'] is False
    assert any(np.allclose(sol, s, atol=1e-8)
               for s in basis_solutions(preconditioner, Cm, rhs_e))

def test_sbls_perturbed(terminate):
    # with C_s, K_G for preconditioner 12 has the wrong inertia (more than m
    # negative eigenvalues), so G is perturbed
    sol, rhs_c, inform = solve(rhs_e, C=C_s, preconditioner=12)
    assert inform['status'] == 0
    assert inform['perturbed'] is True
    # the computed solution still satisfies A x - C y = b
    assert A @ sol[:n] - C_s @ sol[n:] == pytest.approx(rhs_e[n:], abs=1e-8)

def test_sbls_wrong_inertia(terminate):
    # as above, but without perturbation (status -20, which is a warning)
    with pytest.warns(RuntimeWarning):
        factorize(C=C_s, preconditioner=12, perturb=False)
    assert sbls.information()['status'] == -20


# 4) the null-space factorization (C = 0), only for preconditioner 2
NULL_SPACE_PR = 'https://github.com/ralna/GALAHAD/pull/656'

# the null-space factorization writes out of bounds when H_21 is nonzero,
# and ignores the H_11 and H_21 of dense, sparse-by-rows and diagonal H,
# which may corrupt memory, so this cannot be a strict xfail
@pytest.mark.skip(reason='the sbls null-space factorization gives wrong '
                  f'solutions (see {NULL_SPACE_PR})')
@pytest.mark.parametrize('H_type', ['coordinate', 'sparse_by_rows', 'dense'])
def test_sbls_null_space(terminate, H_type):
    sol, rhs_c, inform = solve(rhs_e, factorization=3, H_type=H_type)
    assert inform['status'] == 0
    assert inform['factorization'] == 3
    assert sol == pytest.approx(sol_e, abs=1e-8)

@pytest.mark.parametrize('preconditioner', [1, 3, 5])
def test_sbls_null_space_preconditioner(terminate, preconditioner):
    # the null-space factorization is only available for preconditioner 2
    # (status -45, which is a warning)
    with pytest.warns(RuntimeWarning):
        sbls.load(n, m, 'coordinate', *matrix_storage(H, 'coordinate',
                                                      True)[:4],
                  'coordinate', *matrix_storage(A, 'coordinate', False)[:4],
                  'zero', 0, None, None, None, options(preconditioner, 3))
        H_val = matrix_storage(H, 'coordinate', True)[4]
        A_val = matrix_storage(A, 'coordinate', False)[4]
        sbls.factorize_matrix(n, m, len(H_val), H_val, len(A_val), A_val,
                              0, None)
    assert sbls.information()['status'] == -45

def test_sbls_null_space_c_nonzero(terminate):
    # with C nonzero, another factorization is used
    rhs = K(H, C_s) @ z_star
    sol, rhs_c, inform = solve(rhs, C=C_s, factorization=3)
    assert inform['status'] == 0
    assert inform['factorization'] != 3
    assert sol == pytest.approx(z_star, abs=1e-8)


# 5) storage schemes, with G = H
H_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']
A_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']
C_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

@pytest.mark.parametrize('C_type', C_STORAGE)
@pytest.mark.parametrize('A_type', A_STORAGE)
@pytest.mark.parametrize('H_type', H_STORAGE)
def test_sbls_storage(terminate, H_type, A_type, C_type):
    rhs = K(H, C_s) @ z_star
    sol, rhs_c, inform = solve(rhs, C=C_s, H_type=H_type, A_type=A_type,
                               C_type=C_type)
    assert inform['status'] == 0
    assert sol == pytest.approx(z_star, abs=1e-8)

# H and C given by the schemes without indices: H = diag( 1, 2, 3 ), 2 I,
# I (H = 0 would make K_H singular, as A has a null space), and
# C = diag( 2, 3 ), 2 I, I, 0
H_TRIVIAL = [('diagonal', np.diag([1.0, 2.0, 3.0])),
             ('scaled_identity', 2.0 * np.eye(n)),
             ('identity', np.eye(n))]
C_TRIVIAL = [('diagonal', np.diag([2.0, 3.0])),
             ('scaled_identity', 2.0 * np.eye(m)),
             ('identity', np.eye(m)), ('zero', np.zeros((m, m))),
             ('none', np.zeros((m, m)))]

@pytest.mark.parametrize('H_type, H_t', H_TRIVIAL,
                         ids=[h[0] for h in H_TRIVIAL])
def test_sbls_h_trivial(terminate, H_type, H_t):
    rhs = K(H_t, C_s) @ z_star
    sol, rhs_c, inform = solve(rhs, H=H_t, C=C_s, H_type=H_type)
    assert inform['status'] == 0
    assert sol == pytest.approx(z_star, abs=1e-8)

@pytest.mark.parametrize('C_type, C_t', C_TRIVIAL,
                         ids=[c[0] for c in C_TRIVIAL])
def test_sbls_c_trivial(terminate, C_type, C_t):
    rhs = K(H, C_t) @ z_star
    sol, rhs_c, inform = solve(rhs, C=C_t, C_type=C_type)
    assert inform['status'] == 0
    assert sol == pytest.approx(z_star, abs=1e-8)


def test_sbls_refactorize(terminate):
    # problem 2) with G = H and C_s, after a factorization with C = 0
    H_ne, H_row, H_col, H_ptr, H_val = matrix_storage(H, 'coordinate', True)
    A_ne, A_row, A_col, A_ptr, A_val = matrix_storage(A, 'coordinate', False)
    C_ne, C_row, C_col, C_ptr, C_val = matrix_storage(C_s, 'coordinate', True)
    sbls.load(n, m, 'coordinate', H_ne, H_row, H_col, H_ptr,
              'coordinate', A_ne, A_row, A_col, A_ptr,
              'coordinate', C_ne, C_row, C_col, C_ptr, options())
    sbls.factorize_matrix(n, m, H_ne, H_val, A_ne, A_val, C_ne,
                          np.zeros(C_ne))
    sol = sbls.solve_system(n, m, rhs_e.copy())
    assert sol == pytest.approx(sol_e, abs=1e-8)
    sbls.factorize_matrix(n, m, H_ne, H_val, A_ne, A_val, C_ne, C_val)
    rhs = K(H, C_s) @ z_star
    sol = sbls.solve_system(n, m, rhs.copy())
    assert sbls.information()['status'] == 0
    assert sol == pytest.approx(z_star, abs=1e-8)
