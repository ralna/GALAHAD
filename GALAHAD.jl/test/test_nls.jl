# test_nls.jl
# Simple code to test the Julia interface to NLS

using GALAHAD
using Test
using Printf
using Accessors

# Custom userdata struct
struct userdata_nls
  p::Float64
end

function test_nls()
  # compute the residuals
  function res(n::Int, m::Int, x::Vector{Float64}, c::Vector{Float64},
               userdata::userdata_nls)
    c[1] = x[1]^2 + userdata.p
    c[2] = x[1] + x[2]^2
    c[3] = x[1] - x[2]
    return 0
  end

  # compute the Jacobian
  function jac(n::Int, m::Int, jne::Int, x::Vector{Float64}, jval::Vector{Float64},
               userdata::userdata_nls)
    jval[1] = 2.0 * x[1]
    jval[2] = 1.0
    jval[3] = 2.0 * x[2]
    jval[4] = 1.0
    jval[5] = -1.0
    return 0
  end

  # compute the Hessian
  function hess(n::Int, m::Int, hne::Int, x::Vector{Float64}, y::Vector{Float64},
                hval::Vector{Float64}, userdata::userdata_nls)
    hval[1] = 2.0 * y[1]
    hval[2] = 2.0 * y[2]
    return 0
  end

  # compute Jacobian-vector products
  function jacprod(n::Int, m::Int, x::Vector{Float64}, trans::Bool, u::Vector{Float64},
                   v::Vector{Float64}, got_j::Bool, userdata::userdata_nls)
    if trans
      u[1] = u[1] + 2.0 * x[1] * v[1] + v[2] + v[3]
      u[2] = u[2] + 2.0 * x[2] * v[2] - v[3]
    else
      u[1] = u[1] + 2.0 * x[1] * v[1]
      u[2] = u[2] + v[1] + 2.0 * x[2] * v[2]
      u[3] = u[3] + v[1] - v[2]
    end
    return 0
  end

  # compute Hessian-vector products
  function hessprod(n::Int, m::Int, x::Vector{Float64}, y::Vector{Float64},
                    u::Vector{Float64}, v::Vector{Float64}, got_h::Bool,
                    userdata::userdata_nls)
    u[1] = u[1] + 2.0 * y[1] * v[1]
    u[2] = u[2] + 2.0 * y[2] * v[2]
    return 0
  end

  # compute residual-Hessians-vector products
  function rhessprods(n::Int, m::Int, pne::Int, x::Vector{Float64}, v::Vector{Float64},
                      pval::Vector{Float64}, got_h::Bool, userdata::userdata_nls)
    pval[1] = 2.0 * v[1]
    pval[2] = 2.0 * v[2]
    return 0
  end

  # # scale v
  function scale(n::Int, m::Int, x::Vector{Float64}, u::Vector{Float64}, v::Vector{Float64},
                 userdata::userdata_nls)
    u[1] = v[1]
    u[2] = v[2]
    return 0
  end

  # compute the dense Jacobian
  function jac_dense(n::Int, m::Int, jne::Int, x::Vector{Float64}, jval::Vector{Float64},
                     userdata::userdata_nls)
    jval[1] = 2.0 * x[1]
    jval[2] = 0.0
    jval[3] = 1.0
    jval[4] = 2.0 * x[2]
    jval[5] = 1.0
    jval[6] = -1.0
    return 0
  end

  # compute the dense Hessian
  function hess_dense(n::Int, m::Int, hne::Int, x::Vector{Float64}, y::Vector{Float64},
                      hval::Vector{Float64}, userdata::userdata_nls)
    hval[1] = 2.0 * y[1]
    hval[2] = 0.0
    hval[3] = 2.0 * y[2]
    return 0
  end

  # compute dense residual-Hessians-vector products
  function rhessprods_dense(n::Int, m::Int, pne::Int, x::Vector{Float64},
                            v::Vector{Float64}, pval::Vector{Float64}, got_h::Bool,
                            userdata::userdata_nls)
    pval[1] = 2.0 * v[1]
    pval[2] = 0.0
    pval[3] = 0.0
    pval[4] = 2.0 * v[2]
    pval[5] = 0.0
    pval[6] = 0.0
    return 0
  end

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{nls_control_type{Float64}}()
  inform = Ref{nls_inform_type{Float64}}()

  # Set user data
  userdata = userdata_nls(1.0)

  # Set problem data
  n = 2 # # variables
  m = 3 # # residuals
  j_ne = 5 # Jacobian elements
  h_ne = 2 # Hesssian elements
  p_ne = 2 # residual-Hessians-vector products elements
  J_row = Cint[1, 2, 2, 3, 3]  # Jacobian J
  J_col = Cint[1, 1, 2, 1, 2]  #
  J_ptr = Cint[1, 2, 4, 6]  # row pointers
  H_row = Cint[1, 2]  # Hessian H
  H_col = Cint[1, 2]  # NB lower triangle
  H_ptr = Cint[1, 2, 3]  # row pointers
  P_row = Cint[1, 2]  # residual-Hessians-vector product matrix
  P_ptr = Cint[1, 2, 3, 3]  # column pointers

  # Set storage
  g = zeros(Float64, n) # gradient
  c = zeros(Float64, m) # residual
  y = zeros(Float64, m) # multipliers
  W = Float64[1.0, 1.0, 1.0]  # weights
  st = ' '
  status = Ref{Cint}()

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" tests reverse-communication options\n\n")

  # reverse-communication input/output
  eval_status = Ref{Cint}()
  u = zeros(Float64, max(m, n))
  v = zeros(Float64, max(m, n))
  J_val = zeros(Float64, j_ne)
  J_dense = zeros(Float64, m * n)
  H_val = zeros(Float64, h_ne)
  H_dense = zeros(Float64, div(n * (n + 1), 2))
  H_diag = zeros(Float64, n)
  P_val = zeros(Float64, p_ne)
  P_dense = zeros(Float64, m * n)
  trans = Ref{Bool}()
  got_j = false
  got_h = false

  for d in 1:5
    # Initialize NLS
    nls_initialize(data, control, inform)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing
    # @reset control[].print_level = Cint(1)
    @reset control[].jacobian_available = Cint(2)
    @reset control[].hessian_available = Cint(2)
    @reset control[].model = Cint(6)
    x = Float64[1.5, 1.5]  # starting point
    W = Float64[1.0, 1.0, 1.0]  # weights

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      nls_import(control, data, status, n, m,
                 "coordinate", j_ne, J_row, J_col, C_NULL,
                 "coordinate", h_ne, H_row, H_col, C_NULL,
                 "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
      terminated = false
      while !terminated # reverse-communication loop
        nls_solve_reverse_with_mat(data, status, eval_status,
                                   n, m, x, c, g, j_ne, J_val, y,
                                   h_ne, H_val, v, p_ne, P_val)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate c
          eval_status[] = res(n, m, x, c, userdata)
        elseif status[] == 3 # evaluate J
          eval_status[] = jac(n, m, j_ne, x, J_val, userdata)
        elseif status[] == 4 # evaluate H
          eval_status[] = hess(n, m, h_ne, x, y, H_val, userdata)
        elseif status[] == 7 # evaluate P
          eval_status[] = rhessprods(n, m, p_ne, x, v, P_val, got_h, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # sparse by rows
    if d == 2
      st = 'R'
      nls_import(control, data, status, n, m,
                 "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                 "sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
                 "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

      terminated = false
      while !terminated # reverse-communication loop
        nls_solve_reverse_with_mat(data, status, eval_status,
                                   n, m, x, c, g, j_ne, J_val, y,
                                   h_ne, H_val, v, p_ne, P_val)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate c
          eval_status[] = res(n, m, x, c, userdata)
        elseif status[] == 3 # evaluate J
          eval_status[] = jac(n, m, j_ne, x, J_val, userdata)
        elseif status[] == 4 # evaluate H
          eval_status[] = hess(n, m, h_ne, x, y, H_val, userdata)
        elseif status[] == 7 # evaluate P
          eval_status[] = rhessprods(n, m, p_ne, x, v, P_val, got_h, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # dense
    if d == 3
      st = 'D'
      nls_import(control, data, status, n, m,
                 "dense", j_ne, C_NULL, C_NULL, C_NULL,
                 "dense", h_ne, C_NULL, C_NULL, C_NULL,
                 "dense", p_ne, C_NULL, C_NULL, C_NULL, W)

      terminated = false
      while !terminated # reverse-communication loop
        nls_solve_reverse_with_mat(data, status, eval_status,
                                   n, m, x, c, g, m * n, J_dense, y,
                                   n * (n + 1) / 2, H_dense, v, m * n,
                                   P_dense)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate c
          eval_status[] = res(n, m, x, c, userdata)
        elseif status[] == 3 # evaluate J
          eval_status[] = jac_dense(n, m, j_ne, x, J_dense, userdata)
        elseif status[] == 4 # evaluate H
          eval_status[] = hess_dense(n, m, h_ne, x, y, H_dense, userdata)
        elseif status[] == 7 # evaluate P
          eval_status[] = rhessprods_dense(n, m, p_ne, x, v, P_dense, got_h, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # diagonal
    if d == 4
      st = 'I'
      nls_import(control, data, status, n, m,
                 "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                 "diagonal", h_ne, C_NULL, C_NULL, C_NULL,
                 "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

      terminated = false
      while !terminated # reverse-communication loop
        nls_solve_reverse_with_mat(data, status, eval_status,
                                   n, m, x, c, g, j_ne, J_val, y,
                                   n, H_diag, v, p_ne, P_val)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate c
          eval_status[] = res(n, m, x, c, userdata)
        elseif status[] == 3 # evaluate J
          eval_status[] = jac(n, m, j_ne, x, J_val, userdata)
        elseif status[] == 4 # evaluate H
          eval_status[] = hess(n, m, h_ne, x, y, H_diag, userdata)
        elseif status[] == 7 # evaluate P
          eval_status[] = rhessprods(n, m, p_ne, x, v, P_val, got_h, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # access by products
    if d == 5
      st = 'P'
      # @reset control[].print_level = Cint(1)
      nls_import(control, data, status, n, m,
                 "absent", j_ne, C_NULL, C_NULL, C_NULL,
                 "absent", h_ne, C_NULL, C_NULL, C_NULL,
                 "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

      terminated = false
      while !terminated # reverse-communication loop
        nls_solve_reverse_without_mat(data, status, eval_status,
                                      n, m, x, c, g, trans,
                                      u, v, y, p_ne, P_val)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate c
          eval_status[] = res(n, m, x, c, userdata)
        elseif status[] == 5 # evaluate u + J v or u + J'v
          eval_status[] = jacprod(n, m, x, trans[], u, v, got_j, userdata)
        elseif status[] == 6 # evaluate u + H v
          eval_status[] = hessprod(n, m, x, y, u, v, got_h, userdata)
        elseif status[] == 7 # evaluate P
          eval_status[] = rhessprods(n, m, p_ne, x, v, P_val, got_h, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    nls_information(data, inform, status)

    if inform[].status == 0
      @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
              st, inform[].iter, inform[].obj, inform[].status)
    else
      @printf("%c: NLS_solve exit status = %1i\n", st, inform[].status)
    end

    # Delete internal workspace
    nls_terminate(data, control, inform)
  end

  @printf("\n basic tests of models used, reverse access\n\n")
  for model in 3:8
    # Initialize NLS
    nls_initialize(data, control, inform)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing
    # @reset control[].print_level = Cint(1)
    @reset control[].jacobian_available = Cint(2)
    @reset control[].hessian_available = Cint(2)
    @reset control[].model = Cint(model)
    x = Float64[1.5, 1.5]  # starting point
    W = Float64[1.0, 1.0, 1.0]  # weights

    nls_import(control, data, status, n, m,
               "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
               "sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
               "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

    terminated = false
    while !terminated # reverse-communication loop
      nls_solve_reverse_with_mat(data, status, eval_status,
                                 n, m, x, c, g, j_ne, J_val, y,
                                 h_ne, H_val, v, p_ne, P_val)
      if status[] == 0 # successful termination
        terminated = true
      elseif status[] < 0 # error exit
        terminated = true
      elseif status[] == 2 # evaluate c
        eval_status[] = res(n, m, x, c, userdata)
      elseif status[] == 3 # evaluate J
        eval_status[] = jac(n, m, j_ne, x, J_val, userdata)
      elseif status[] == 4 # evaluate H
        eval_status[] = hess(n, m, h_ne, x, y, H_val, userdata)
      elseif status[] == 7 # evaluate P
        eval_status[] = rhessprods(n, m, p_ne, x, v, P_val, got_h, userdata)
      else
        @printf(" the value %1i of status should not occur\n", status)
      end
    end

    nls_information(data, inform, status)

    if inform[].status == 0
      @printf("P%1i:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
              model, inform[].iter, inform[].obj, inform[].status)
    else
      @printf(" %i: NLS_solve exit status = %1i\n", model, inform[].status)
    end

    # Delete internal workspace
    nls_terminate(data, control, inform)
  end

  @printf("\n basic tests of models used, reverse access by products\n\n")
  for model in 3:8
    # Initialize NLS
    nls_initialize(data, control, inform)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing
    # @reset control[].print_level = 1
    @reset control[].jacobian_available = Cint(2)
    @reset control[].hessian_available = Cint(2)
    @reset control[].model = Cint(model)
    x = Float64[1.5, 1.5]  # starting point
    W = Float64[1.0, 1.0, 1.0]  # weights

    nls_import(control, data, status, n, m,
               "absent", j_ne, C_NULL, C_NULL, C_NULL,
               "absent", h_ne, C_NULL, C_NULL, C_NULL,
               "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

    terminated = false
    while !terminated # reverse-communication loop
      nls_solve_reverse_without_mat(data, status, eval_status,
                                    n, m, x, c, g, trans,
                                    u, v, y, p_ne, P_val)
      if status[] == 0 # successful termination
        terminated = true
      elseif status[] < 0 # error exit
        terminated = true
      elseif status[] == 2 # evaluate c
        eval_status[] = res(n, m, x, c, userdata)
      elseif status[] == 5 # evaluate u + J v or u + J'v
        eval_status[] = jacprod(n, m, x, trans[], u, v, got_j, userdata)
      elseif status[] == 6 # evaluate u + H v
        eval_status[] = hessprod(n, m, x, y, u, v, got_h, userdata)
      elseif status[] == 7 # evaluate P
        eval_status[] = rhessprods(n, m, p_ne, x, v, P_val, got_h, userdata)
      else
        @printf(" the value %1i of status should not occur\n", status)
      end
    end

    nls_information(data, inform, status)

    if inform[].status == 0
      @printf("P%1i:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
              model, inform[].iter, inform[].obj, inform[].status)
    else
      @printf("P%i: NLS_solve exit status = %1i\n", model, inform[].status)
    end

    # Delete internal workspace
    nls_terminate(data, control, inform)
  end

  return 0
end

@testset "NLS" begin
  @test test_nls() == 0
end
