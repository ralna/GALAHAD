# test_expo.jl
# Simple code to test the Julia interface to EXPO

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

# Custom userdata struct
mutable struct userdata_expo{T}
  p::T
  pass_userdata::Bool
  eval_fc::Function
  eval_gj::Function
  eval_hl::Function
  eval_fc_ptr::Ptr{Cvoid}
  eval_gj_ptr::Ptr{Cvoid}
  eval_hl_ptr::Ptr{Cvoid}
end

function test_expo(::Type{T}, ::Type{INT}; mode::String="direct", sls::String="sytr") where {T,INT}

  # compute the objective and constraints
  function eval_fc(x::Vector{T}, f::Vector{T}, c::Vector{T}, userdata::userdata_expo{T})
    f[1] = x[1]^2 + x[2]^2
    c[1] = x[1] + x[2] - 1
    c[2] = x[1]^2 + x[2]^2 - 1
    c[3] = userdata.p * x[1]^2 + x[2]^2 - userdata.p
    c[4] = x[1]^2 - x[2]
    c[5] = x[2]^2 - x[1]
    return f, c
  end

  # compute the gradient and Jacobian
  function eval_gj(x::Vector{T}, g::Vector{T}, jval::Vector{T}, userdata::userdata_expo{T})
    g[1] = 2 * x[1]
    g[2] = 2 * x[2]
    jval[1] = 1
    jval[2] = 1
    jval[3] = 2 * x[1]
    jval[4] = 2 * x[2]
    jval[5] = 2 * userdata.p * x[1]
    jval[6] = 2 * x[2]
    jval[7] = 2 * x[1]
    jval[8] = -1
    jval[9] = -1
    jval[10] = 2 * x[2]
    return g, jval
  end

  # compute the gradient and dense Jacobian
  function eval_gj_dense(x::Vector{T}, g::Vector{T}, jval::Vector{T}, userdata::userdata_expo{T})
    g[1] = 2 * x[1]
    g[2] = 2 * x[2]
    jval[1] = 1
    jval[2] = 1
    jval[3] = 2 * x[1]
    jval[4] = 2 * x[2]
    jval[5] = 2 * userdata.p * x[1]
    jval[6] = 2 * x[2]
    jval[7] = 2 * x[1]
    jval[8] = -1
    jval[9] = -1
    jval[10] = 2 * x[2]
    return g, jval
  end

  # compute the Hessian
  function eval_hl(x::Vector{T}, y::Vector{T}, hval::Vector{T}, userdata::userdata_expo{T})
    hval[1] = 2 - 2 * (y[2] + userdata.p * y[3] + y[4])
    hval[2] = 2 - 2 * (y[2] + y[3] + y[5])
    return hval
  end

  # compute the dense Hessian
  function eval_hl_dense(x::Vector{T}, y::Vector{T}, hval::Vector{T}, userdata::userdata_expo{T})
    hval[1] = 2 - 2 * (y[2] + userdata.p * y[3] + y[4])
    hval[2] = 0
    hval[3] = 2 - 2* (y[2] + y[3] + y[5])
    return hval
  end

  # Pointer to C-compatible functions that wraps the Julia functions for
  # objective, contraints, gradient, jacobian and hessian of the problem
  eval_fc_ptr = galahad_fc(T, INT)
  eval_gj_ptr = galahad_gj(T, INT)
  eval_hl_ptr = galahad_hl(T, INT)

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{expo_control_type{T,INT}}()
  inform = Ref{expo_inform_type{T,INT}}()

  # Set user data
  use_userdata = true
  userdata = userdata_expo{T}(9, use_userdata, eval_fc, eval_gj, eval_hl, eval_fc_ptr, eval_gj_ptr, eval_hl_ptr)
  userdata_ptr = pointer_from_objref(userdata)
  userdata_dense = userdata_expo{T}(9, use_userdata, eval_fc, eval_gj_dense, eval_hl_dense, eval_fc_ptr, eval_gj_ptr, eval_hl_ptr)
  userdata_dense_ptr = pointer_from_objref(userdata_dense)

  # Set problem data
  n = INT(2)  # variables
  m = INT(5)  # constraints
  j_ne = INT(10) # Jacobian elements
  h_ne = INT(2)  # Hesssian elements
  j_ne_dense = INT(10) # dense Jacobian elements
  h_ne_dense = INT(3) # dense Jacobian elements
  J_row = INT[1, 1, 2, 2, 3, 3, 4, 4, 5, 5]  # Jacobian J
  J_col = INT[1, 2, 1, 2, 1, 2, 1, 2, 1, 2]  #
  J_ptr = INT[1, 3, 5, 7, 9, 11]  # row pointers
  H_row = INT[1, 2]  # Hessian H
  H_col = INT[1, 2]  # NB lower triangle
  H_ptr = INT[1, 2, 3]  # row pointers

  # Set storage
  y = zeros(T, m)  # multipliers
  z = zeros(T, n)  # dual variables
  c = zeros(T, m)  # constraints
  gl = zeros(T, n) # gradient
  x_l = T[-50.0, -50.0]  # variable lower bound
  x_u = T[50.0, 50.0]  # variable upper bound
  c_l = T[0.0, 0.0, 0.0, 0.0, 0.0]  # constraint lower bound
  c_u = T[Inf, Inf, Inf, Inf, Inf]  # constraint upper bound
  st = ' '
  status = Ref{INT}()

  @printf(" Fortran sparse matrix indexing\n\n")

  if mode == "direct"
    @printf(" test direct-communication options\n\n")

    for d in 1:4
      # Initialize EXPO
      expo_initialize(T, INT, data, control, inform)

      # Set linear solvers
      @reset control[].ssls_control.symmetric_linear_solver = galahad_linear_solver(sls)

      # Set user-defined control options
      # @reset control[].print_level = INT(10)
      # @reset control[].tru_control.print_level = INT(10)
      # @reset control[].ssls_control.print_level = INT(10)
      # @reset control[].ssls_control.sls_control.print_level = INT(10)

      @reset control[].max_it = INT(20)
      @reset control[].max_eval = INT(100)
      @reset control[].stop_abs_p = T(0.00001)
      @reset control[].stop_abs_d = T(0.00001)
      @reset control[].stop_abs_c = T(0.00001)

      x = T[3.0, 1.0]  # starting point

      # sparse co-ordinate storage
      if d == 1
        st = 'C'
        expo_import(T, INT, control, data, status, n, m,
                    "coordinate", j_ne, J_row, J_col, C_NULL,
                    "coordinate", h_ne, H_row, H_col, C_NULL )

        expo_solve_hessian_direct(T, INT, data,
                                  userdata_ptr, status, n, m, j_ne, h_ne,
                                  c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                  eval_fc_ptr, eval_gj_ptr, eval_hl_ptr)
      end

      # sparse by rows
      if d == 2
        st = 'R'
        expo_import(T, INT, control, data, status, n, m,
                    "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                    "sparse_by_rows", h_ne, C_NULL, H_col, H_ptr )

        expo_solve_hessian_direct(T, INT, data,
                                  userdata_ptr, status, n, m, j_ne, h_ne,
                                  c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                  eval_fc_ptr, eval_gj_ptr, eval_hl_ptr)
      end

      # dense
      if d == 3
        st = 'D'
        expo_import(T, INT, control, data, status, n, m,
                    "dense", j_ne_dense, C_NULL, C_NULL, C_NULL,
                    "dense", h_ne_dense, C_NULL, C_NULL, C_NULL )

        expo_solve_hessian_direct(T, INT, data,
                                  userdata_dense_ptr, status, n, m, j_ne_dense, h_ne_dense,
                                  c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                  eval_fc_ptr, eval_gj_ptr, eval_hl_ptr)
      end

      # diagonal
      if d == 4
        st = 'I'
        expo_import(T, INT, control, data, status, n, m,
                    "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                    "diagonal", n, C_NULL, C_NULL, C_NULL )

        expo_solve_hessian_direct(T, INT, data,
                                  userdata_ptr, status, n, m, j_ne, n,
                                  c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                  eval_fc_ptr, eval_gj_ptr, eval_hl_ptr)
      end

      expo_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
                st, inform[].iter, inform[].obj, inform[].status)
      else
        @printf("%c: EXPO_solve exit status = %1i\n", st, inform[].status)
      end

      # Delete internal workspace
      expo_terminate(T, INT, data, control, inform)
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
    @testset "EXPO -- $T -- $INT" begin
      @testset "$mode communication" for mode in ("direct",)
        @test test_expo(T, INT; mode) == 0
      end
    end
  end
end
