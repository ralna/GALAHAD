# test_nls.jl
# Simple code to test the Julia interface to NLS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

# Custom userdata struct
mutable struct userdata_nls{T}
  p::T
end

function test_nls(::Type{T}, ::Type{INT}; mode::String="reverse", sls::String="sytr", dls::String="potr") where {T,INT}

  # compute the residuals
  function res(x::Vector{T}, c::Vector{T}, userdata::userdata_nls{T})
    c[1] = x[1]^2 + userdata.p
    c[2] = x[1] + x[2]^2
    c[3] = x[1] - x[2]
    return INT(0)
  end

  function res_c(n::INT, m::INT, x::Ptr{T}, c::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _c = unsafe_wrap(Vector{T}, c, m)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_nls{T}
    res(_x, _c, _userdata)
  end

  res_ptr = @eval @cfunction($res_c, $INT, ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # compute the Jacobian
  function jac(x::Vector{T}, jval::Vector{T}, userdata::userdata_nls{T})
    jval[1] = 2.0 * x[1]
    jval[2] = 1.0
    jval[3] = 2.0 * x[2]
    jval[4] = 1.0
    jval[5] = -1.0
    return INT(0)
  end

  function jac_c(n::INT, m::INT, jne::INT, x::Ptr{T}, jval::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _jval = unsafe_wrap(Vector{T}, jval, jne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_nls{T}
    jac(_x, _jval, _userdata)
  end

  jac_ptr = @eval @cfunction($jac_c, $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # compute the Hessian
  function hess(x::Vector{T}, y::Vector{T}, hval::Vector{T},
                userdata::userdata_nls{T})
    hval[1] = 2.0 * y[1]
    hval[2] = 2.0 * y[2]
    return INT(0)
  end

  function hess_c(n::INT, m::INT, hne::INT, x::Ptr{T}, y::Ptr{T}, hval::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _y = unsafe_wrap(Vector{T}, y, m)
    _hval = unsafe_wrap(Vector{T}, hval, hne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_nls{T}
    hess(_x, _y, _hval, _userdata)
  end

  hess_ptr = @eval @cfunction($hess_c, $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # compute Jacobian-vector products
  function jacprod(x::Vector{T}, transpose::Bool, u::Vector{T}, v::Vector{T},
                   got_j::Bool, userdata::userdata_nls{T})
    if transpose
      u[1] = u[1] + 2.0 * x[1] * v[1] + v[2] + v[3]
      u[2] = u[2] + 2.0 * x[2] * v[2] - v[3]
    else
      u[1] = u[1] + 2.0 * x[1] * v[1]
      u[2] = u[2] + v[1] + 2.0 * x[2] * v[2]
      u[3] = u[3] + v[1] - v[2]
    end
    return INT(0)
  end

  function jacprod_c(n::INT, m::INT, x::Ptr{T}, transpose::Bool, u::Ptr{T}, v::Ptr{T}, got_j::Bool, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _u = unsafe_wrap(Vector{T}, u, transpose ? n : m)
    _v = unsafe_wrap(Vector{T}, v, transpose ? m : n)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_nls{T}
    jacprod(_x, transpose, _u, _v, got_j, _userdata)
  end

  jacprod_ptr = @eval @cfunction($jacprod_c, $INT, ($INT, $INT, Ptr{$T}, Bool, Ptr{$T}, Ptr{$T}, Bool, Ptr{Cvoid}))

  # compute Hessian-vector products
  function hessprod(x::Vector{T}, y::Vector{T}, u::Vector{T}, v::Vector{T},
                    got_h::Bool, userdata::userdata_nls{T})
    u[1] = u[1] + 2.0 * y[1] * v[1]
    u[2] = u[2] + 2.0 * y[2] * v[2]
    return INT(0)
  end

  function hessprod_c(n::INT, m::INT, x::Ptr{T}, y::Ptr{T}, u::Ptr{T}, v::Ptr{T}, got_h::Bool, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _y = unsafe_wrap(Vector{T}, y, m)
    _u = unsafe_wrap(Vector{T}, u, n)
    _v = unsafe_wrap(Vector{T}, v, n)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_nls{T}
    hessprod(_x, _y, _u, _v, got_h, _userdata)
  end

  hessprod_ptr = @eval @cfunction($hessprod_c, $INT, ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Bool, Ptr{Cvoid}))

  # compute residual-Hessians-vector products
  function rhessprods(x::Vector{T}, v::Vector{T}, pval::Vector{T},
                      got_h::Bool, userdata::userdata_nls{T})
    pval[1] = 2.0 * v[1]
    pval[2] = 2.0 * v[2]
    return INT(0)
  end

  function rhessprods_c(n::INT, m::INT, pne::INT, x::Ptr{T}, v::Ptr{T}, pval::Ptr{T}, got_h::Bool, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _v = unsafe_wrap(Vector{T}, v, n)
    _pval = unsafe_wrap(Vector{T}, pval, pne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_nls{T}
    rhessprods(_x, _v, _pval, got_h, _userdata)
  end

  rhessprods_ptr = @eval @cfunction($rhessprods_c, $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Bool, Ptr{Cvoid}))

  # compute the dense Jacobian
  function jac_dense(x::Vector{T}, jval::Vector{T}, userdata::userdata_nls{T})
    jval[1] = 2.0 * x[1]
    jval[2] = 0.0
    jval[3] = 1.0
    jval[4] = 2.0 * x[2]
    jval[5] = 1.0
    jval[6] = -1.0
    return INT(0)
  end

  function jac_dense_c(n::INT, m::INT, jne::INT, x::Ptr{T}, jval::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _jval = unsafe_wrap(Vector{T}, jval, jne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_nls{T}
    jac_dense(_x, _jval, _userdata)
  end

  jac_dense_ptr = @eval @cfunction($jac_dense_c, $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # compute the dense Hessian
  function hess_dense(x::Vector{T}, y::Vector{T}, hval::Vector{T},
                      userdata::userdata_nls{T})
    hval[1] = 2.0 * y[1]
    hval[2] = 0.0
    hval[3] = 2.0 * y[2]
    return INT(0)
  end

  function hess_dense_c(n::INT, m::INT, hne::INT, x::Ptr{T}, y::Ptr{T}, hval::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _y = unsafe_wrap(Vector{T}, y, m)
    _hval = unsafe_wrap(Vector{T}, hval, hne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_nls{T}
    hess_dense(_x, _y, _hval, _userdata)
  end

  hess_dense_ptr = @eval @cfunction($hess_dense_c, $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # compute dense residual-Hessians-vector products
  function rhessprods_dense(x::Vector{T}, v::Vector{T}, pval::Vector{T},
                            got_h::Bool, userdata::userdata_nls{T})
    pval[1] = 2.0 * v[1]
    pval[2] = 0.0
    pval[3] = 0.0
    pval[4] = 2.0 * v[2]
    pval[5] = 0.0
    pval[6] = 0.0
    return INT(0)
  end

  function rhessprods_dense_c(n::INT, m::INT, pne::INT, x::Ptr{T}, v::Ptr{T}, pval::Ptr{T}, got_h::Bool, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _v = unsafe_wrap(Vector{T}, v, n)
    _pval = unsafe_wrap(Vector{T}, pval, pne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_nls{T}
    rhessprods_dense(_x, _v, _pval, got_h, _userdata)
  end

  rhessprods_dense_ptr = @eval @cfunction($rhessprods_dense_c, $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Bool, Ptr{Cvoid}))

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{nls_control_type{T,INT}}()
  inform = Ref{nls_inform_type{T,INT}}()

  # Set user data
  userdata = userdata_nls{T}(1)
  userdata_ptr = pointer_from_objref(userdata)

  # Set problem data
  n = INT(2)  # variables
  m = INT(3)  # residuals
  j_ne = INT(5)  # Jacobian elements
  h_ne = INT(2)  # Hesssian elements
  p_ne = INT(2)  # residual-Hessians-vector products elements
  j_ne_dense = INT(6)  # dense Jacobian elements
  h_ne_dense = INT(3)  # dense Hesssian elements
  p_ne_dense = INT(6)  # dense residual-Hessians-vector
  J_row = INT[1, 2, 2, 3, 3]  # Jacobian J
  J_col = INT[1, 1, 2, 1, 2]  #
  J_ptr = INT[1, 2, 4, 6]  # row pointers
  H_row = INT[1, 2]  # Hessian H
  H_col = INT[1, 2]  # NB lower triangle
  H_ptr = INT[1, 2, 3]  # row pointers
  P_row = INT[1, 2]  # residual-Hessians-vector product matrix
  P_ptr = INT[1, 2, 3, 3]  # column pointers

  # Set storage
  g = zeros(T, n) # gradient
  c = zeros(T, m) # residual
  y = zeros(T, m) # multipliers
  W = T[1.0, 1.0, 1.0]  # weights
  st = ' '
  status = Ref{INT}()

  @printf(" Fortran sparse matrix indexing\n\n")

  if mode == "direct"
    @printf(" tests options for all-in-one storage format\n\n")

    for d in 1:5
      # Initialize NLS
      nls_initialize(T, INT, data, control, inform)

      # Set linear solvers
      @reset control[].psls_control.definite_linear_solver = galahad_linear_solver(dls)
      @reset control[].rqs_control.symmetric_linear_solver = galahad_linear_solver(sls)
      @reset control[].rqs_control.definite_linear_solver = galahad_linear_solver(dls)

      # Set user-defined control options
      # @reset control[].print_level = INT(1)
      @reset control[].jacobian_available = INT(2)
      @reset control[].hessian_available = INT(2)
      @reset control[].model = INT(6)
      x = T[1.5, 1.5]  # starting point
      W = T[1.0, 1.0, 1.0]  # weights

      # sparse co-ordinate storage
      if d == 1
        st = 'C'
        nls_import(T, INT, control, data, status, n, m,
                   "coordinate", j_ne, J_row, J_col, C_NULL,
                   "coordinate", h_ne, H_row, H_col, C_NULL,
                   "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        nls_solve_with_mat(T, INT, data, userdata_ptr, status,
                           n, m, x, c, g, res_ptr, j_ne, jac_ptr,
                           h_ne, hess_ptr, p_ne, rhessprods_ptr)
      end

      # sparse by rows
      if d == 2
        st = 'R'
        nls_import(T, INT, control, data, status, n, m,
                   "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                   "sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
                   "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        nls_solve_with_mat(T, INT, data, userdata_ptr, status,
                           n, m, x, c, g, res_ptr, j_ne, jac_ptr,
                           h_ne, hess_ptr, p_ne, rhessprods_ptr)
      end

      # dense
      if d == 3
        st = 'D'
        nls_import(T, INT, control, data, status, n, m,
                   "dense", j_ne_dense, C_NULL, C_NULL, C_NULL,
                   "dense", h_ne_dense, C_NULL, C_NULL, C_NULL,
                   "dense", p_ne_dense, C_NULL, C_NULL, C_NULL, W)

        nls_solve_with_mat(T, INT, data, userdata_ptr, status,
                           n, m, x, c, g, res_ptr, j_ne_dense, jac_dense_ptr,
                           h_ne_dense, hess_dense_ptr, p_ne_dense, rhessprods_dense_ptr)
      end

      # diagonal
      if d == 4
        st = 'I'
        nls_import(T, INT, control, data, status, n, m,
                   "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                   "diagonal", n, C_NULL, C_NULL, C_NULL,
                   "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        nls_solve_with_mat(T, INT, data, userdata_ptr, status,
                           n, m, x, c, g, res_ptr, j_ne, jac_ptr,
                           n, hess_ptr, p_ne, rhessprods_ptr)
      end

      # access by products
      if d == 5
        st = 'P'
        nls_import(T, INT, control, data, status, n, m,
                   "absent", j_ne, C_NULL, C_NULL, C_NULL,
                   "absent", h_ne, C_NULL, C_NULL, C_NULL,
                   "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        nls_solve_without_mat(T, INT, data, userdata_ptr, status,
                              n, m, x, c, g, res_ptr, jacprod_ptr,
                              hessprod_ptr, p_ne, rhessprods_ptr)
      end

      nls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("%c:%6i iterations. Optimal objective value = %.2f status = %1i\n",
                st, inform[].iter, inform[].obj, inform[].status)
      else
        @printf("%c: NLS_solve exit status = %1i\n", st, inform[].status)
      end

      # Delete internal workspace
      nls_terminate(T, INT, data, control, inform)
    end
  end

  if mode == "reverse"
    @printf(" tests reverse-communication options\n\n")

    # reverse-communication input/output
    eval_status = Ref{INT}()
    u = zeros(T, max(m, n))
    v = zeros(T, max(m, n))
    J_val = zeros(T, j_ne)
    J_dense = zeros(T, m * n)
    H_val = zeros(T, h_ne)
    H_dense = zeros(T, div(n * (n + 1), 2))
    H_diag = zeros(T, n)
    P_val = zeros(T, p_ne)
    P_dense = zeros(T, m * n)
    trans = Ref{Bool}()
    got_j = false
    got_h = false

    for d in 1:5
      # Initialize NLS
      nls_initialize(T, INT, data, control, inform)

      # Linear solvers
      @reset control[].psls_control.definite_linear_solver = galahad_linear_solver(dls)
      @reset control[].rqs_control.symmetric_linear_solver = galahad_linear_solver(sls)
      @reset control[].rqs_control.definite_linear_solver = galahad_linear_solver(dls)

      # Set user-defined control options
      # @reset control[].print_level = INT(1)
      @reset control[].jacobian_available = INT(2)
      @reset control[].hessian_available = INT(2)
      @reset control[].model = INT(6)
      x = T[1.5, 1.5]  # starting point
      W = T[1.0, 1.0, 1.0]  # weights

      # sparse co-ordinate storage
      if d == 1
        st = 'C'
        nls_import(T, INT, control, data, status, n, m,
                   "coordinate", j_ne, J_row, J_col, C_NULL,
                   "coordinate", h_ne, H_row, H_col, C_NULL,
                   "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
        terminated = false
        while !terminated # reverse-communication loop
          nls_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                     n, m, x, c, g, j_ne, J_val, y,
                                     h_ne, H_val, v, p_ne, P_val)
          if status[] == 0 # successful termination
            terminated = true
          elseif status[] < 0 # error exit
            terminated = true
          elseif status[] == 2 # evaluate c
            eval_status[] = res(x, c, userdata)
          elseif status[] == 3 # evaluate J
            eval_status[] = jac(x, J_val, userdata)
          elseif status[] == 4 # evaluate H
            eval_status[] = hess(x, y, H_val, userdata)
          elseif status[] == 7 # evaluate P
            eval_status[] = rhessprods(x, v, P_val, got_h, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      # sparse by rows
      if d == 2
        st = 'R'
        nls_import(T, INT, control, data, status, n, m,
                   "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                   "sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
                   "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        terminated = false
        while !terminated # reverse-communication loop
          nls_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                     n, m, x, c, g, j_ne, J_val, y,
                                     h_ne, H_val, v, p_ne, P_val)
          if status[] == 0 # successful termination
            terminated = true
          elseif status[] < 0 # error exit
            terminated = true
          elseif status[] == 2 # evaluate c
            eval_status[] = res(x, c, userdata)
          elseif status[] == 3 # evaluate J
            eval_status[] = jac(x, J_val, userdata)
          elseif status[] == 4 # evaluate H
            eval_status[] = hess(x, y, H_val, userdata)
          elseif status[] == 7 # evaluate P
            eval_status[] = rhessprods(x, v, P_val, got_h, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      # dense
      if d == 3
        st = 'D'
        nls_import(T, INT, control, data, status, n, m,
                   "dense", j_ne_dense, C_NULL, C_NULL, C_NULL,
                   "dense", h_ne_dense, C_NULL, C_NULL, C_NULL,
                   "dense", p_ne_dense, C_NULL, C_NULL, C_NULL, W)

        terminated = false
        while !terminated # reverse-communication loop
          nls_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                     n, m, x, c, g, j_ne_dense, J_dense, y,
                                     h_ne_dense, H_dense, v, p_ne_dense,
                                     P_dense)
          if status[] == 0 # successful termination
            terminated = true
          elseif status[] < 0 # error exit
            terminated = true
          elseif status[] == 2 # evaluate c
            eval_status[] = res(x, c, userdata)
          elseif status[] == 3 # evaluate J
            eval_status[] = jac_dense(x, J_dense, userdata)
          elseif status[] == 4 # evaluate H
            eval_status[] = hess_dense(x, y, H_dense, userdata)
          elseif status[] == 7 # evaluate P
            eval_status[] = rhessprods_dense(x, v, P_dense, got_h, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      # diagonal
      if d == 4
        st = 'I'
        nls_import(T, INT, control, data, status, n, m,
                   "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                   "diagonal", n, C_NULL, C_NULL, C_NULL,
                   "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        terminated = false
        while !terminated # reverse-communication loop
          nls_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                     n, m, x, c, g, j_ne, J_val, y,
                                     n, H_diag, v, p_ne, P_val)
          if status[] == 0 # successful termination
            terminated = true
          elseif status[] < 0 # error exit
            terminated = true
          elseif status[] == 2 # evaluate c
            eval_status[] = res(x, c, userdata)
          elseif status[] == 3 # evaluate J
            eval_status[] = jac(x, J_val, userdata)
          elseif status[] == 4 # evaluate H
            eval_status[] = hess(x, y, H_diag, userdata)
          elseif status[] == 7 # evaluate P
            eval_status[] = rhessprods(x, v, P_val, got_h, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      # access by products
      if d == 5
        st = 'P'
        # @reset control[].print_level = INT(1)
        nls_import(T, INT, control, data, status, n, m,
                   "absent", j_ne, C_NULL, C_NULL, C_NULL,
                   "absent", h_ne, C_NULL, C_NULL, C_NULL,
                   "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        terminated = false
        while !terminated # reverse-communication loop
          nls_solve_reverse_without_mat(T, INT, data, status, eval_status,
                                        n, m, x, c, g, trans,
                                        u, v, y, p_ne, P_val)
          if status[] == 0 # successful termination
            terminated = true
          elseif status[] < 0 # error exit
            terminated = true
          elseif status[] == 2 # evaluate c
            eval_status[] = res(x, c, userdata)
          elseif status[] == 5 # evaluate u + J v or u + J'v
            eval_status[] = jacprod(x, trans[], u, v, got_j, userdata)
          elseif status[] == 6 # evaluate u + H v
            eval_status[] = hessprod(x, y, u, v, got_h, userdata)
          elseif status[] == 7 # evaluate P
            eval_status[] = rhessprods(x, v, P_val, got_h, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      nls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
                st, inform[].iter, inform[].obj, inform[].status)
      else
        @printf("%c: NLS_solve exit status = %1i\n", st, inform[].status)
      end

      # Delete internal workspace
      nls_terminate(T, INT, data, control, inform)
    end
  end

  if mode == "direct"
    @printf("\n basic tests of models used, direct access\n\n")

    for model in 3:8
      # Initialize NLS
      nls_initialize(T, INT, data, control, inform)

      # Linear solvers
      @reset control[].psls_control.definite_linear_solver = galahad_linear_solver(dls)
      @reset control[].rqs_control.symmetric_linear_solver = galahad_linear_solver(sls)
      @reset control[].rqs_control.definite_linear_solver = galahad_linear_solver(dls)

      # Set user-defined control options
      # @reset control[].print_level = INT(1)
      @reset control[].jacobian_available = INT(2)
      @reset control[].hessian_available = INT(2)
      @reset control[].model = INT(model)
      x = T[1.5, 1.5]  # starting point
      W = T[1.0, 1.0, 1.0]  # weights

      nls_import(T, INT, control, data, status, n, m,
                 "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                 "sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
                 "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

      nls_solve_with_mat(T, INT, data, userdata_ptr, status,
                         n, m, x, c, g, res_ptr, j_ne, jac_ptr,
                         h_ne, hess_ptr, p_ne, rhessprods_ptr)

      nls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("P%1i: %6i iterations. Optimal objective value = %.2f, status = %1i\n",
                model, inform[].iter, inform[].obj, inform[].status)
      else
        @printf("%i: NLS_solve exit status = %1i\n", model, inform[].status)
      end

      # Delete internal workspace
      nls_terminate(T, INT, data, control, inform)
    end

    @printf("\n basic tests of models used, access by products\n\n")

    for model in 3:8
      # Initialize NLS
      nls_initialize(T, INT, data, control, inform)

      # Linear solvers
      @reset control[].psls_control.definite_linear_solver = galahad_linear_solver(dls)
      @reset control[].rqs_control.symmetric_linear_solver = galahad_linear_solver(sls)
      @reset control[].rqs_control.definite_linear_solver = galahad_linear_solver(dls)

      # Set user-defined control options
      # @reset control[].print_level = INT(1)
      @reset control[].jacobian_available = INT(2)
      @reset control[].hessian_available = INT(2)
      @reset control[].model = INT(model)
      x = T[1.5,1.5]  # starting point
      W = T[1.0, 1.0, 1.0]  # weights

      nls_import(T, INT, control, data, status, n, m,
                 "absent", j_ne, C_NULL, C_NULL, C_NULL,
                 "absent", h_ne, C_NULL, C_NULL, C_NULL,
                 "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

      nls_solve_without_mat(T, INT, data, userdata_ptr, status,
                            n, m, x, c, g, res_ptr, jacprod_ptr,
                            hessprod_ptr, p_ne, rhessprods_ptr)

      nls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("P%1i :%6i iterations. Optimal objective value = %.2f status = %1i\n",
                model, inform[].iter, inform[].obj, inform[].status)
      else
        @printf("P%i: NLS_solve exit status = %1i\n", model, inform[].status)
      end

      # Delete internal workspace
      nls_terminate(T, INT, data, control, inform)
    end
  end

  if mode == "reverse"
    @printf("\n basic tests of models used, reverse access\n\n")
    for model in 3:8
      # Initialize NLS
      nls_initialize(T, INT, data, control, inform)

      # Linear solvers
      @reset control[].psls_control.definite_linear_solver = galahad_linear_solver(dls)
      @reset control[].rqs_control.symmetric_linear_solver = galahad_linear_solver(sls)
      @reset control[].rqs_control.definite_linear_solver = galahad_linear_solver(dls)

      # Set user-defined control options
      # @reset control[].print_level = INT(1)
      @reset control[].jacobian_available = INT(2)
      @reset control[].hessian_available = INT(2)
      @reset control[].model = INT(model)
      x = T[1.5, 1.5]  # starting point
      W = T[1.0, 1.0, 1.0]  # weights

      nls_import(T, INT, control, data, status, n, m,
                 "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                 "sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
                 "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

      terminated = false
      while !terminated # reverse-communication loop
        nls_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                   n, m, x, c, g, j_ne, J_val, y,
                                   h_ne, H_val, v, p_ne, P_val)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate c
          eval_status[] = res(x, c, userdata)
        elseif status[] == 3 # evaluate J
          eval_status[] = jac(x, J_val, userdata)
        elseif status[] == 4 # evaluate H
          eval_status[] = hess(x, y, H_val, userdata)
        elseif status[] == 7 # evaluate P
          eval_status[] = rhessprods(x, v, P_val, got_h, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end

      nls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("P%1i:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
                model, inform[].iter, inform[].obj, inform[].status)
      else
        @printf(" %i: NLS_solve exit status = %1i\n", model, inform[].status)
      end

      # Delete internal workspace
      nls_terminate(T, INT, data, control, inform)
    end

    @printf("\n basic tests of models used, reverse access by products\n\n")
    for model in 3:8
      # Initialize NLS
      nls_initialize(T, INT, data, control, inform)

      # Linear solvers
      @reset control[].psls_control.definite_linear_solver = galahad_linear_solver(dls)
      @reset control[].rqs_control.symmetric_linear_solver = galahad_linear_solver(sls)
      @reset control[].rqs_control.definite_linear_solver = galahad_linear_solver(dls)

      # Set user-defined control options
      # @reset control[].print_level = 1
      @reset control[].jacobian_available = INT(2)
      @reset control[].hessian_available = INT(2)
      @reset control[].model = INT(model)
      x = T[1.5, 1.5]  # starting point
      W = T[1.0, 1.0, 1.0]  # weights

      nls_import(T, INT, control, data, status, n, m,
                 "absent", j_ne, C_NULL, C_NULL, C_NULL,
                 "absent", h_ne, C_NULL, C_NULL, C_NULL,
                 "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

      terminated = false
      while !terminated # reverse-communication loop
        nls_solve_reverse_without_mat(T, INT, data, status, eval_status,
                                      n, m, x, c, g, trans,
                                      u, v, y, p_ne, P_val)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate c
          eval_status[] = res(x, c, userdata)
        elseif status[] == 5 # evaluate u + J v or u + J'v
          eval_status[] = jacprod(x, trans[], u, v, got_j, userdata)
        elseif status[] == 6 # evaluate u + H v
          eval_status[] = hessprod(x, y, u, v, got_h, userdata)
        elseif status[] == 7 # evaluate P
          eval_status[] = rhessprods(x, v, P_val, got_h, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end

      nls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("P%1i:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
                model, inform[].iter, inform[].obj, inform[].status)
      else
        @printf("P%i: NLS_solve exit status = %1i\n", model, inform[].status)
      end

      # Delete internal workspace
      nls_terminate(T, INT, data, control, inform)
    end
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
    @testset "NLS -- $T -- $INT" begin
      @testset "$mode communication" for mode in ("reverse", "direct")
        @test test_nls(T, INT; mode) == 0
      end
    end
  end
end
