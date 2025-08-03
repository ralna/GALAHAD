# test_bnls.jl
# Simple code to test the Julia interface to BNLS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

# Custom userdata struct
mutable struct userdata_bnls{T}
  p::T
end

function test_bnls(::Type{T}, ::Type{INT}; mode::String="reverse") where {T,INT}
  # compute the residuals
  function res(x::Vector{T}, c::Vector{T}, userdata::userdata_bnls{T})
    p = userdata.p
    c[1] = x[1] * x[1] + p
    c[2] = x[1] + x[2] * x[2]
    c[3] = x[1] - x[2]
    return INT(0)
  end

  # compute the Jacobian
  function jac(x::Vector{T}, jval::Vector{T}, userdata::userdata_bnls{T})
    jval[1] = 2 * x[1]
    jval[2] = 1
    jval[3] = 2 * x[2]
    jval[4] = 1
    jval[5] = -1
    return INT(0)
  end

  # compute the Hessian
  function hess(x::Vector{T}, y::Vector{T}, hval::Vector{T}, userdata::userdata_bnls{T})
    hval[1] = 2 * y[1]
    hval[2] = 2 * y[1]
    return INT(0)
  end

  # compute Jacobian-vector products
  function jacprod(x::Vector{T}, transpose::Bool, u::Vector{T}, v::Vector{T}, got_j::Bool,
                   userdata::userdata_bnls{T})
    if transpose
      u[1] = u[1] + 2 * x[1] * v[1] + v[2] + v[3]
      u[2] = u[2] + 2 * x[2] * v[2] - v[3]
    else
      u[1] = u[1] + 2 * x[1] * v[1]
      u[2] = u[2] + v[1] + 2 * x[2] * v[]
      u[3] = u[3] + v[1] - v[2]
    end
    return INT(0)
  end

  # compute Hessian-vector products
  function hessprod(x::Vector{T}, y::Vector{T}, u::Vector{T}, v::Vector{T}, got_h::Bool,
                    userdata::userdata_bnls{T})
    u[1] = u[1] + 2 * y[1] * v[1]
    u[2] = u[2] + 2 * y[2] * v[2]
    return INT(0)
  end

  # compute residual-Hessians-vector products
  function rhessprods(x::Vector{T}, v::Vector{T}, pval::Vector{T}, got_h::Bool,
                      userdata::userdata_bnls{T})
    pval[1] = 2 * v[1]
    pval[2] = 2 * v[2]
    return INT(0)
  end

  # scale v
  function scale(x::Vector{T}, u::Vector{T}, v::Vector{T}, userdata::userdata_bnls{T})
    u[1] = v[1]
    u[2] = v[2]
    return INT(0)
  end

  # compute the dense Jacobian
  function jac_dense(x::Vector{T}, jval::Vector{T}, userdata::userdata_bnls{T})
    jval[1] = 2 * x[1]
    jval[2] = 0
    jval[3] = 1
    jval[4] = 2 * x[2]
    jval[5] = 1
    jval[6] = -1
    return INT(0)
  end

  # compute the dense Hessian
  function hess_dense(x::Vector{T}, y::Vector{T}, hval::Vector{T},
                      userdata::userdata_bnls{T})
    hval[1] = 2 * y[1]
    hval[2] = 0
    hval[3] = 2 * y[2]
    return INT(0)
  end

  # compute dense residual-Hessians-vector products
  function rhessprods_dense(x::Vector{T}, v::Vector{T}, pval::Vector{T}, got_h::Bool,
                            userdata::userdata_bnls{T})
    pval[1] = 2 * v[1]
    pval[2] = 0
    pval[3] = 0
    pval[4] = 2 * v[2]
    pval[5] = 0
    pval[6] = 0
    return INT(0)
  end

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{bnls_control_type{T,INT}}()
  inform = Ref{bnls_inform_type{T,INT}}()

  # Set user data
  userdata = userdata_bnls{T}(1)

  # Set problem data
  n = INT(2)  # variables
  m = INT(3)  # residuals
  j_ne = INT(5)  # Jacobian elements
  h_ne = INT(2)  # Hesssian elements
  p_ne = INT(2)  # residual-Hessians-vector products elements
  J_row = INT[1, 2, 2, 3, 3]  # Jacobian J
  J_col = INT[1, 1, 2, 1, 2]  #
  J_ptr = INT[1, 2, 4, 6]  # row pointers
  H_row = INT[1, 2]  # Hessian H
  H_col = INT[1, 2]  # NB lower triangle
  H_ptr = INT[1, 2, 3]  # row pointers
  P_row = INT[1, 2]  # residual-Hessians-vector product matrix
  P_ptr = INT[1, 2, 3, 3]  # column pointers

  # Set storage
  g = zeros(T, n)  # gradient
  c = zeros(T, m)  # residual
  y = zeros(T, m)  # multipliers
  st = ' '
  status = Ref{INT}(0)

  @printf(" Fortran sparse matrix indexing\n\n")

  if mode == "direct"
    @printf(" tests options for all-in-one storage format\n\n")

    for d in 1:5
      # Initialize BNLS
      bnls_initialize(T, INT, data, control, inform)

      # @reset control[].print_level = 1
      @reset control[].jacobian_available = 2
      @reset control[].hessian_available = 2
      @reset control[].model = 6
      x = T[1.5, 1.5]  # starting point
      W = T[1.0, 1.0, 1.0]  # weights

      # sparse co-ordinate storage
      if d == 1
        st = 'C'
        bnls_import(T, INT, control, data, status, n, m,
                    "coordinate", j_ne, J_row, J_col, C_NULL,
                    "coordinate", h_ne, H_row, H_col, C_NULL,
                    "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        bnls_solve_with_mat(T, INT, data, userdata, status,
                            n, m, x, c, g, res, j_ne, jac,
                            h_ne, hess, p_ne, rhessprods)
      end

      # sparse by rows
      if d == 2
        st = 'R'
        bnls_import(T, INT, control, data, status, n, m,
                    "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                    "sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
                    "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        bnls_solve_with_mat(T, INT, data, userdata, status,
                            n, m, x, c, g, res, j_ne, jac,
                            h_ne, hess, p_ne, rhessprods)
      end

      # dense
      if d == 3
        st = 'D'
        bnls_import(T, INT, control, data, status, n, m,
                    "dense", j_ne, C_NULL, C_NULL, C_NULL,
                    "dense", h_ne, C_NULL, C_NULL, C_NULL,
                    "dense", p_ne, C_NULL, C_NULL, C_NULL, W)

        bnls_solve_with_mat(T, INT, data, userdata, status,
                            n, m, x, c, g, res, j_ne, jac_dense,
                            h_ne, hess_dense, p_ne, rhessprods_dense)
      end
      # diagonal
      if d == 4
        st = 'I'
        bnls_import(T, INT, control, data, status, n, m,
                    "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                    "diagonal", h_ne, C_NULL, C_NULL, C_NULL,
                    "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        bnls_solve_with_mat(T, INT, data, userdata, status,
                            n, m, x, c, g, res, j_ne, jac,
                            h_ne, hess, p_ne, rhessprods)
      end

      # access by products
      if d == 5
        st = 'P'
        bnls_import(T, INT, control, data, status, n, m,
                    "absent", j_ne, C_NULL, C_NULL, C_NULL,
                    "absent", h_ne, C_NULL, C_NULL, C_NULL,
                    "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        bnls_solve_without_mat(T, INT, data, userdata, status,
                               n, m, x, c, g, res, jacprod,
                               hessprod, p_ne, rhessprods)
      end

      bnls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("%c:%6i iterations. Optimal objective value = %5.2f, status = %1i\n",
                st, inform[].iter, inform[].obj, inform[].status)
      else
        @printf("%c: BNLS_solve exit status = %1i\n", st, inform[].status)
      end

      # Delete internal workspace
      bnls_terminate(T, INT, data, control, inform)
    end
  end

  if mode == "reverse"
    @printf("\n tests reverse-communication options\n\n")

    # reverse-communication input / output
    eval_status = Ref{INT}()
    u = zeros(T, min(m, n))
    v = zeros(T, min(m, n))
    J_val = zeros(T, j_ne)
    J_dense = zeros(T, m * n)
    H_val = zeros(T, h_ne)
    H_dense = zeros(T, div(n * (n + 1), 2))
    H_diag = zeros(T, n)
    P_val = zeros(T, p_ne)
    P_dense = zeros(T, m * n)
    got_j = false
    got_h = false

    for d in 1:5
      # Initialize BNLS
      bnls_initialize(T, INT, data, control, inform)

      # Set user-defined control options
      # @reset control[].print_level = 1
      @reset control[].jacobian_available = 2
      @reset control[].hessian_available = 2
      @reset control[].model = 6
      x = T[1.5, 1.5]  # starting point
      W = T[1.0, 1.0, 1.0]  # weights

      # sparse co-ordinate storage
      if d == 1
        st = 'C'
        bnls_import(T, INT, control, data, status, n, m,
                    "coordinate", j_ne, J_row, J_col, C_NULL,
                    "coordinate", h_ne, H_row, H_col, C_NULL,
                    "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        terminated = false
        while !terminated  # reverse-communication loop
          bnls_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                      n, m, x, c, g, j_ne, J_val, y,
                                      h_ne, H_val, v, p_ne, P_val)

          if status[] == 0 # successful termination
            terminated = true
          elseif status[] < 0  # error exit
            terminated = true
          elseif status[] == 2  # evaluate c
            eval_status[] = res(x, c, userdata)
          elseif status[] == 3  # evaluate J
            eval_status[] = jac(x, J_val, userdata)
          elseif status[] == 4  # evaluate H
            eval_status[] = hess(x, y, H_val, userdata)
          elseif status[] == 7  # evaluate P
            eval_status[] = rhessprods(x, v, P_val, got_h, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      # sparse by rows
      if d == 2
        st = 'R'
        bnls_import(T, INT, control, data, status, n, m,
                    "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                    "sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
                    "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        terminated = false
        while !terminated  # reverse-communication loop
          bnls_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                      n, m, x, c, g, j_ne, J_val, y,
                                      h_ne, H_val, v, p_ne, P_val)

          if status[] == 0  # successful termination
            terminated = true
          elseif status[] < 0  # error exit
            terminated = true
          elseif status[] == 2  # evaluate c
            eval_status[] = res(x, c, userdata)
          elseif status[] == 3  # evaluate J
            eval_status[] = jac(j_ne, x, J_val, userdata)
          elseif status[] == 4  # evaluate H
            eval_status[] = hess(h_ne, x, y, H_val, userdata)
          elseif status[] == 7  # evaluate P
            eval_status[] = rhessprods(p_ne, x, v, P_val, got_h, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      # dense
      if d == 3
        st = 'D'
        bnls_import(T, INT, control, data, status, n, m,
                    "dense", j_ne, C_NULL, C_NULL, C_NULL,
                    "dense", h_ne, C_NULL, C_NULL, C_NULL,
                    "dense", p_ne, C_NULL, C_NULL, C_NULL, W)

        terminated = false
        while !terminated  # reverse-communication loop
          bnls_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                      n, m, x, c, g, m * n, J_dense, y,
                                      n * (n + 1) / 2, H_dense, v, m * n,
                                      P_dense)

          if status[] == 0  # successful termination
            terminated = true
          elseif status[] < 0  # error exit
            terminated = true
          elseif status[] == 2  # evaluate c
            eval_status[] = res(x, c, userdata)
          elseif status[] == 3  # evaluate J
            eval_status[] = jac_dense(x, J_dense, userdata)
          elseif status[] == 4  # evaluate H
            eval_status[] = hess_dense(x, y, H_dense, userdata)
          elseif status[] == 7  # evaluate P
            eval_status[] = rhessprods_dense(x, v, P_dense, got_h, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      # diagonal
      if d == 4
        st = 'I'
        bnls_import(T, INT, control, data, status, n, m,
                    "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                    "diagonal", h_ne, C_NULL, C_NULL, C_NULL,
                    "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        terminated = false
        while !terminated  # reverse-communication loop
          bnls_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                      n, m, x, c, g, j_ne, J_val, y,
                                      n, H_diag, v, p_ne, P_val)

          if status[] == 0  # successful termination
            terminated = true
          elseif status[] < 0  # error exit
            terminated = true
          elseif status[] == 2  # evaluate c
            eval_status[] = res(x, c, userdata)
          elseif status[] == 3  # evaluate J
            eval_status[] = jac(x, J_val, userdata)
          elseif status[] == 4  # evaluate H
            eval_status[] = hess(x, y, H_diag, userdata)
          elseif status[] == 7  # evaluate P
            eval_status[] = rhessprods(x, v, P_val, got_h, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      # access by products
      if d == 5
        st = 'P'
        bnls_import(T, INT, control, data, status, n, m,
                    "absent", j_ne, C_NULL, C_NULL, C_NULL,
                    "absent", h_ne, C_NULL, C_NULL, C_NULL,
                    "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

        terminated = false
        while !terminated  # reverse-communication loop
          bnls_solve_reverse_without_mat(data, status, eval_status,
                                         n, m, x, c, g, transpose,
                                         u, v, y, p_ne, P_val)
          if status[] == 0  # successful termination
            terminated = true
          elseif status < 0  # error exit
            terminated = true
          elseif status[] == 2  # evaluate c
            eval_status[] = res(x, c, userdata)
          elseif status[] == 5  # evaluate u + J v or u + J'v
            eval_status[] = jacprod(x, transpose, u, v, got_j, userdata)
          elseif status[] == 6  # evaluate u + H v
            eval_status[] = hessprod(x, y, u, v, got_h, userdata)
          elseif status[] == 7  # evaluate P
            eval_status[] = rhessprods(x, v, P_val, got_h, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      bnls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("%c:%6i iterations. Optimal objective value = %5.2f, status = %1i\n",
                st, inform[].iter, inform[].obj, inform[].status)
      else
        @printf("%c: BNLS_solve exit status = %1i\n", st, inform[].status)
      end

      # Delete internal workspace
      bnls_terminate(T, INT, data, control, inform)
    end
  end

  if mode == "direct"
    @printf("\n basic tests of models used, direct access\n\n")

    for model in 3:8
      # Initialize BNLS
      bnls_initialize(T, INT, data, control, inform)

      # Set user-defined control options
      # @reset control[].print_level = 1
      @reset control[].jacobian_available = 2
      @reset control[].hessian_available = 2
      @reset control[].model = model
      x = T[1.5, 1.5]  # starting point
      W = T[1.0, 1.0, 1.0]  # weights

      bnls_import(T, INT, control, data, status, n, m,
                  "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                  "sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
                  "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

      bnls_solve_with_mat(T, INT, data, userdata, status,
                          n, m, x, c, g, res, j_ne, jac,
                          h_ne, hess, p_ne, rhessprods)

      bnls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf(" %1i:%6i iterations. Optimal objective value = %5.2f, status = %1i\n",
                model, inform[].iter, inform[].obj, inform[].status)
      else
        @printf(" %i: BNLS_solve exit status = %1i\n", model, inform[].status)
      end

      # Delete internal workspace
      bnls_terminate(T, INT, data, control, inform)
    end

    @printf("\n basic tests of models used, access by products\n\n")

    for model in 3:8
      # Initialize BNLS
      bnls_initialize(T, INT, data, control, inform)

      # Set user-defined control options
      # @reset control[].print_level = 1
      @reset control[].jacobian_available = 2
      @reset control[].hessian_available = 2
      @reset control[].model = model
      x = T[1.5, 1.5]  # starting point
      W = T[1.0, 1.0, 1.0]  # weights

      bnls_import(T, INT, control, data, status, n, m,
                  "absent", j_ne, C_NULL, C_NULL, C_NULL,
                  "absent", h_ne, C_NULL, C_NULL, C_NULL,
                  "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

      bnls_solve_without_mat(T, INT, data, userdata, status,
                             n, m, x, c, g, res, jacprod,
                             hessprod, p_ne, rhessprods)

      bnls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("P%1i:%6i iterations. Optimal objective value = %5.2f, status = %1i\n",
                model, inform[].iter, inform[].obj, inform[].status)
      else
        @printf("P%i: BNLS_solve exit status = %1i\n", model, inform[].status)
      end

      # Delete internal workspace
      bnls_terminate(T, INT, data, control, inform)
    end
  end

  if mode == "reverse"
    @printf("\n basic tests of models used, reverse access\n\n")

    for model in 3:8
      # Initialize BNLS
      bnls_initialize(T, INT, data, control, inform)

      # Set user-defined control options
      # @reset control[].print_level = 1
      @reset control[].jacobian_available = 2
      @reset control[].hessian_available = 2
      @reset control[].model = model
      x = T[1.5, 1.5]  # starting point
      W = T[1.0, 1.0, 1.0]  # weights

      bnls_import(T, INT, control, data, status, n, m,
                  "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                  "sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
                  "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

      terminated = false
      while !terminated  # reverse-communication loop
        bnls_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                    n, m, x, c, g, j_ne, J_val, y,
                                    h_ne, H_val, v, p_ne, P_val)

        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0  # error exit
          terminated = true
        elseif status[] == 2  # evaluate c
          eval_status[] = res(x, c, userdata)
        elseif status[] == 3  # evaluate J
          eval_status[] = jac(x, J_val, userdata)
        elseif status[] == 4  # evaluate H
          eval_status[] = hess(x, y, H_val, userdata)
        elseif status[] == 7  # evaluate P
          eval_status[] = rhessprods(x, v, P_val, got_h, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end

      bnls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("P%1i:%6i iterations. Optimal objective value = %5.2f, status = %1i\n",
                model, inform[].iter, inform[].obj, inform[].status)
      else
        @printf(" %i: BNLS_solve exit status = %1i\n", model, inform[].status)
      end

      # Delete internal workspace
      bnls_terminate(T, INT, data, control, inform)
    end

    @printf("\n basic tests of models used, reverse access by products\n\n")

    for model in 3:8
      # Initialize BNLS
      bnls_initialize(T, INT, data, control, inform)

      # Set user-defined control options
      # @reset control[].print_level = 1
      @reset control[].jacobian_available = 2
      @reset control[].hessian_available = 2
      @reset control[].model = model
      x = T[1.5, 1.5]  # starting point
      W = T[1.0, 1.0, 1.0]  # weights

      bnls_import(T, INT, control, data, status, n, m,
                  "absent", j_ne, C_NULL, C_NULL, C_NULL,
                  "absent", h_ne, C_NULL, C_NULL, C_NULL,
                  "sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)

      terminated = false
      while !terminated  # reverse-communication loop
        bnls_solve_reverse_without_mat(T, INT, data, status, eval_status,
                                       n, m, x, c, g, transpose,
                                       u, v, y, p_ne, P_val)

        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0  # error exit
          terminated = true
        elseif status[] == 2  # evaluate c
          eval_status[] = res(x, c, userdata)
        elseif status[] == 5  # evaluate u + J v or u + J'v
          eval_status[] = jacprod(x, transpose, u, v, got_j, userdata)
        elseif status[] == 6  # evaluate u + H v
          eval_status[] = hessprod(x, y, u, v, got_h, userdata)
        elseif status[] == 7  # evaluate P
          eval_status[] = rhessprods(x, v, P_val, got_h, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end

      bnls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("P%1i:%6i iterations. Optimal objective value = %5.2f, status = %1i\n",
                model, inform[].iter, inform[].obj, inform[].status)
      else
        @printf("P%i: BNLS_solve exit status = %1i\n", model, inform[].status)
      end

      # Delete internal workspace
      bnls_terminate(T, INT, data, control, inform)
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
    @testset "BNLS -- $T -- $INT" begin
      @testset "$mode communication" for mode in ("reverse", "direct")
        @test test_bnls(T, INT; mode) == 0
      end
    end
  end
end
