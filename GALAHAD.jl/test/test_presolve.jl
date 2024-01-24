# test_presolve.jl
# Simple code to test the Julia interface to PRESOLVE

using GALAHAD
using Test
using Printf
using Accessors

function test_presolve()
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{presolve_control_type{Float64}}()
  inform = Ref{presolve_inform_type}()

  # Set problem data
  n = 6 # dimension
  m = 5 # number of general constraints
  H_ne = 1 # Hesssian elements
  H_row = Cint[1]  # row indices, NB lower triangle
  H_col = Cint[1]  # column indices, NB lower triangle
  H_ptr = Cint[1, 2, 2, 2, 2, 2, 2]  # row pointers
  H_val = Float64[1.0]  # values
  g = Float64[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # linear term in the objective
  f = 1.0  # constant term in the objective
  A_ne = 8 # Jacobian elements
  A_row = Cint[3, 3, 3, 4, 4, 5, 5, 5]  # row indices
  A_col = Cint[3, 4, 5, 3, 6, 4, 5, 6]  # column indices
  A_ptr = Cint[1, 1, 1, 4, 6, 9]  # row pointers
  A_val = Float64[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # values
  c_l = Float64[0.0, 0.0, 2.0, 1.0, 3.0]  # constraint lower bound
  c_u = Float64[1.0, 1.0, 3.0, 3.0, 3.0]  # constraint upper bound
  x_l = Float64[-3.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # variable lower bound
  x_u = Float64[3.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # variable upper bound

  # Set output storage
  st = ' '
  status = Ref{Cint}()

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of qp storage formats\n\n")
  for d in 1:7
    n_trans = Ref{Cint}()
    m_trans = Ref{Cint}()
    H_ne_trans = Ref{Cint}()
    A_ne_trans = Ref{Cint}()

    # Initialize PRESOLVE
    presolve_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      presolve_import_problem(control, data, status, n, m,
                              "coordinate", H_ne, H_row, H_col, Cint[], H_val, g, f,
                              "coordinate", A_ne, A_row, A_col, Cint[], A_val,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      presolve_import_problem(control, data, status, n, m,
                              "sparse_by_rows", H_ne, Cint[], H_col, H_ptr, H_val, g, f,
                              "sparse_by_rows", A_ne, Cint[], A_col, A_ptr, A_val,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # dense
    if d == 3
      st = 'D'
      H_dense_ne = div(n * (n + 1), 2) # number of elements of H
      A_dense_ne = m * n # number of elements of A
      H_dense = Float64[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      A_dense = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                        0.0, 1.0, 1.0, 1.0]
      presolve_import_problem(control, data, status, n, m,
                              "dense", H_dense_ne, Cint[], Cint[], Cint[], H_dense, g,
                              f, "dense", A_dense_ne, Cint[], Cint[], Cint[], A_dense,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # diagonal
    if d == 4
      st = 'L'
      presolve_import_problem(control, data, status, n, m,
                              "diagonal", n, Cint[], Cint[], Cint[], H_val, g, f,
                              "sparse_by_rows", A_ne, Cint[], A_col, A_ptr, A_val,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # scaled identity
    if d == 5
      st = 'S'
      presolve_import_problem(control, data, status, n, m,
                              "scaled_identity", 1, Cint[], Cint[], Cint[], H_val, g, f,
                              "sparse_by_rows", A_ne, Cint[], A_col, A_ptr, A_val,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # identity
    if d == 6
      st = 'I'
      presolve_import_problem(control, data, status, n, m,
                              "identity", 0, Cint[], Cint[], Cint[], Cint[], g, f,
                              "sparse_by_rows", A_ne, Cint[], A_col, A_ptr, A_val,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # zero
    if d == 7
      st = 'Z'
      presolve_import_problem(control, data, status, n, m,
                              "zero", 0, Cint[], Cint[], Cint[], Cint[], g, f,
                              "sparse_by_rows", A_ne, Cint[], A_col, A_ptr, A_val,
                              c_l, c_u, x_l, x_u,
                              n_trans, m_trans, H_ne_trans, A_ne_trans)
    end

    # @printf("%c: n, m, h_ne, a_ne = %2i, %2i, %2i, %2i\n", st, n_trans, m_trans, H_ne_trans, A_ne_trans)
    f_trans = Ref{Float64}() # transformed constant term in the objective
    H_ptr_trans = zeros(Cint, n_trans[] + 1) # transformed Hessian row pointers
    H_col_trans = zeros(Cint, H_ne_trans[]) # transformed Hessian column indices
    H_val_trans = zeros(Float64, H_ne_trans[]) # transformed Hessian values
    g_trans = zeros(Float64, n_trans[]) # transformed gradient
    A_ptr_trans = zeros(Cint, m_trans[] + 1) # transformed Jacobian row pointers
    A_col_trans = zeros(Cint, A_ne_trans[]) # transformed Jacobian column indices
    A_val_trans = zeros(Float64, A_ne_trans[]) # transformed Jacobian values
    x_l_trans = zeros(Float64, n_trans[]) # transformed lower variable bounds
    x_u_trans = zeros(Float64, n_trans[]) # transformed upper variable bounds
    c_l_trans = zeros(Float64, m_trans[]) # transformed lower constraint bounds
    c_u_trans = zeros(Float64, m_trans[]) # transformed upper constraint bounds
    y_l_trans = zeros(Float64, m_trans[]) # transformed lower multiplier bounds
    y_u_trans = zeros(Float64, m_trans[]) # transformed upper multiplier bounds
    z_l_trans = zeros(Float64, n_trans[]) # transformed lower dual variable bounds
    z_u_trans = zeros(Float64, n_trans[]) # transformed upper dual variable bounds

    presolve_transform_problem(data, status, n_trans[], m_trans[],
                               H_ne_trans[], H_col_trans, H_ptr_trans,
                               H_val_trans, g_trans, f_trans, A_ne_trans[],
                               A_col_trans, A_ptr_trans, A_val_trans,
                               c_l_trans, c_u_trans, x_l_trans, x_u_trans,
                               y_l_trans, y_u_trans, z_l_trans, z_u_trans)

    x_trans = zeros(Float64, n_trans[]) # transformed variables
    for i in 1:n_trans[]
      x_trans[i] = 0.0
    end
    c_trans = zeros(Float64, m_trans[]) # transformed constraints
    for i in 1:n_trans[]
      c_trans[i] = 0.0
    end
    y_trans = zeros(Float64, m_trans[]) # transformed Lagrange multipliers
    for i in 1:n_trans[]
      y_trans[i] = 0.0
    end
    z_trans = zeros(Float64, n_trans[]) # transformed dual variables
    for i in 1:n_trans[]
      z_trans[i] = 0.0
    end

    x = zeros(Float64, n) # primal variables
    c = zeros(Float64, m) # constraint values
    y = zeros(Float64, m) # Lagrange multipliers
    z = zeros(Float64, n) # dual variables

    # @printf("%c: n_trans, m_trans, n, m = %2i, %2i, %2i, %2i\n", st, n_trans, m_trans, n, m)
    presolve_restore_solution(data, status, n_trans[], m_trans[],
                              x_trans, c_trans, y_trans, z_trans, n, m, x, c, y, z)

    presolve_information(data, inform, status)

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
    presolve_terminate(data, control, inform)
  end

  return 0
end

@testset "PRESOLVE" begin
  @test test_presolve() == 0
end
