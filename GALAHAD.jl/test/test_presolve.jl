# test_presolve.jl
# Simple code to test the Julia interface to PRESOLVE

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_presolve(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{presolve_control_type{T,INT}}()
  inform = Ref{presolve_inform_type{INT}}()

  # Set problem data
  n = INT(6)  # dimension
  m = INT(5)  # number of general constraints
  H_ne = INT(1)  # Hesssian elements
  H_row = INT[1]  # row indices, NB lower triangle
  H_col = INT[1]  # column indices, NB lower triangle
  H_ptr = INT[1, 2, 2, 2, 2, 2, 2]  # row pointers
  H_val = T[1.0]  # values
  g = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # linear term in the objective
  f = one(T)  # constant term in the objective
  A_ne = INT(8)  # Jacobian elements
  A_row = INT[3, 3, 3, 4, 4, 5, 5, 5]  # row indices
  A_col = INT[3, 4, 5, 3, 6, 4, 5, 6]  # column indices
  A_ptr = INT[1, 1, 1, 4, 6, 9]  # row pointers
  A_val = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # values
  c_l = T[0.0, 0.0, 2.0, 1.0, 3.0]  # constraint lower bound
  c_u = T[1.0, 1.0, 3.0, 3.0, 3.0]  # constraint upper bound
  x_l = T[-3.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # variable lower bound
  x_u = T[3.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # variable upper bound

  # Set output storage
  st = ' '
  status = Ref{INT}()

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of qp storage formats\n\n")
  for d in 1:7
    n_trans = Ref{INT}()
    m_trans = Ref{INT}()
    H_ne_trans = Ref{INT}()
    A_ne_trans = Ref{INT}()

    # Initialize PRESOLVE
    presolve_initialize(T, INT, data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      presolve_import_problem(T, INT, control, data, status, n, m,
                              "coordinate", H_ne, H_row, H_col, C_NULL, H_val, g, f,
                              "coordinate", A_ne, A_row, A_col, C_NULL, A_val,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      presolve_import_problem(T, INT, control, data, status, n, m,
                              "sparse_by_rows", H_ne, C_NULL, H_col, H_ptr, H_val, g, f,
                              "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr, A_val,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # dense
    if d == 3
      st = 'D'
      H_dense_ne = div(n * (n + 1), 2) # number of elements of H
      A_dense_ne = m * n # number of elements of A
      H_dense = T[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      A_dense = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                        0.0, 1.0, 1.0, 1.0]
      presolve_import_problem(T, INT, control, data, status, n, m,
                              "dense", H_dense_ne, C_NULL, C_NULL, C_NULL, H_dense, g,
                              f, "dense", A_dense_ne, C_NULL, C_NULL, C_NULL, A_dense,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # diagonal
    if d == 4
      st = 'L'
      presolve_import_problem(T, INT, control, data, status, n, m,
                              "diagonal", n, C_NULL, C_NULL, C_NULL, H_val, g, f,
                              "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr, A_val,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # scaled identity
    if d == 5
      st = 'S'
      presolve_import_problem(T, INT, control, data, status, n, m,
                              "scaled_identity", 1, C_NULL, C_NULL, C_NULL, H_val, g, f,
                              "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr, A_val,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # identity
    if d == 6
      st = 'I'
      presolve_import_problem(T, INT, control, data, status, n, m,
                              "identity", 0, C_NULL, C_NULL, C_NULL, C_NULL, g, f,
                              "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr, A_val,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # zero
    if d == 7
      st = 'Z'
      presolve_import_problem(T, INT, control, data, status, n, m,
                              "zero", 0, C_NULL, C_NULL, C_NULL, C_NULL, g, f,
                              "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr, A_val,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # @printf("%c: n, m, h_ne, a_ne = %2i, %2i, %2i, %2i\n", st, n_trans, m_trans, H_ne_trans, A_ne_trans)
    f_trans = Ref{T}() # transformed constant term in the objective
    H_ptr_trans = zeros(INT, n_trans[] + 1) # transformed Hessian row pointers
    H_col_trans = zeros(INT, H_ne_trans[]) # transformed Hessian column indices
    H_val_trans = zeros(T, H_ne_trans[]) # transformed Hessian values
    g_trans = zeros(T, n_trans[]) # transformed gradient
    A_ptr_trans = zeros(INT, m_trans[] + 1) # transformed Jacobian row pointers
    A_col_trans = zeros(INT, A_ne_trans[]) # transformed Jacobian column indices
    A_val_trans = zeros(T, A_ne_trans[]) # transformed Jacobian values
    x_l_trans = zeros(T, n_trans[]) # transformed lower variable bounds
    x_u_trans = zeros(T, n_trans[]) # transformed upper variable bounds
    c_l_trans = zeros(T, m_trans[]) # transformed lower constraint bounds
    c_u_trans = zeros(T, m_trans[]) # transformed upper constraint bounds
    y_l_trans = zeros(T, m_trans[]) # transformed lower multiplier bounds
    y_u_trans = zeros(T, m_trans[]) # transformed upper multiplier bounds
    z_l_trans = zeros(T, n_trans[]) # transformed lower dual variable bounds
    z_u_trans = zeros(T, n_trans[]) # transformed upper dual variable bounds

    presolve_transform_problem(T, INT, data, status, n_trans[], m_trans[],
                               H_ne_trans[], H_col_trans, H_ptr_trans,
                               H_val_trans, g_trans, f_trans, A_ne_trans[],
                               A_col_trans, A_ptr_trans, A_val_trans,
                               c_l_trans, c_u_trans, x_l_trans, x_u_trans,
                               y_l_trans, y_u_trans, z_l_trans, z_u_trans)

    x_trans = zeros(T, n_trans[]) # transformed variables
    for i in 1:n_trans[]
      x_trans[i] = 0.0
    end
    c_trans = zeros(T, m_trans[]) # transformed constraints
    for i in 1:n_trans[]
      c_trans[i] = 0.0
    end
    y_trans = zeros(T, m_trans[]) # transformed Lagrange multipliers
    for i in 1:n_trans[]
      y_trans[i] = 0.0
    end
    z_trans = zeros(T, n_trans[]) # transformed dual variables
    for i in 1:n_trans[]
      z_trans[i] = 0.0
    end

    x = zeros(T, n) # primal variables
    c = zeros(T, m) # constraint values
    y = zeros(T, m) # Lagrange multipliers
    z = zeros(T, n) # dual variables

    # @printf("%c: n_trans, m_trans, n, m = %2i, %2i, %2i, %2i\n", st, n_trans, m_trans, n, m)
    presolve_restore_solution(T, INT, data, status, n_trans[], m_trans[],
                              x_trans, c_trans, y_trans, z_trans, n, m, x, c, y, z)

    presolve_information(T, INT, data, inform, status)

    if inform[].status == 0
      @printf("%c:%6i transformations, n, m = %2i, %2i, status = %1i\n", st,
              inform[].nbr_transforms, n_trans[], m_trans[], inform[].status)
    else
      @printf("%c: PRESOLVE_solve exit status = %1i\n", st, inform[].status)
    end

    # @printf("x: ")
    # for i = 1:n
    #   @printf("%f ", x[i])
    # end
    # @printf("\n")
    # @printf("gradient: ")
    # for i = 1:n
    #   @printf("%f ", g[i])
    # end
    # @printf("\n")

    # Delete internal workspace
    presolve_terminate(T, INT, data, control, inform)
  end

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "PRESOLVE -- $T -- $INT" begin
      @test test_presolve(T, INT) == 0
    end
  end
end
