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
end

function test_expo(::Type{T}, ::Type{INT}; mode::String="direct", sls::String="sytr") where {T,INT}

  # compute the objective and constraints
  function eval_fc(x::Vector{T}, f::Vector{T}, c::Vector{T}, userdata::userdata_expo{T})
    _f[1] = _x[1]^2 + _x[2]^2
    _c[1] = _x[1] + _x[2] - 1
    _c[2] = _x[1]^2 + _x[2]^2 - 1
    _c[3] = _userdata.p * _x[1]^2 + _x[2]^2 - _userdata.p
    _c[4] = _x[1]^2 - _x[2]
    _c[5] = _x[2]^2 - _x[1]
    return INT(0)
  end

  function eval_fc_c(n::INT, m::INT, x::Ptr{T}, f::Ptr{T}, c::Ptr{T}, userdata::Ptr{Cvoid})
    @assert n == INT(2)
    @assert m == INT(5)
    _x = unsafe_wrap(Vector{T}, x, n)
    _c = unsafe_wrap(Vector{T}, c, m)
    _f = unsafe_wrap(Vector{T}, f, 1)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_expo{T}
    eval_fc(_x, _f, _c, _userdata)
  end

  eval_fc_ptr = @eval @cfunction($eval_fc_c, $INT, ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # compute the gradient and Jacobian
  function eval_gj(x::Vector{T}, g::Vector{T}, jval::Vector{T}, userdata::userdata_expo{T})
    _g[1] = 2 * _x[1]
    _g[2] = 2 * _x[2]
    _jval[1] = 1
    _jval[2] = 1
    _jval[3] = 2 * _x[1]
    _jval[4] = 2 * _x[2]
    _jval[5] = 2 * _userdata.p * _x[1]
    _jval[6] = 2 * _x[2]
    _jval[7] = 2 * _x[1]
    _jval[8] = -1
    _jval[9] = -1
    _jval[10] = 2 * _x[2]
    return INT(0)
  end

  function eval_gj_c(n::INT, m::INT, J_ne::INT, x::Ptr{T}, g::Ptr{T},
                     jval::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _g = unsafe_wrap(Vector{T}, g, n)
    _jval = unsafe_wrap(Vector{T}, jval, J_ne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_expo{T}
    eval_gj(_x, _g, _jval, _userdata)
  end

  eval_gj_ptr = @eval @cfunction($eval_gj_c, $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # compute the gradient and dense Jacobian
  function eval_gj_dense(x::Vector{T}, g::Vector{T}, jval::Vector{T}, userdata::userdata_expo{T})
    _g[1] = 2 * _x[1]
    _g[2] = 2 * _x[2]
    _jval[1] = 1
    _jval[2] = 1
    _jval[3] = 2 * _x[1]
    _jval[4] = 2 * _x[2]
    _jval[5] = 2 * _userdata.p * _x[1]
    _jval[6] = 2 * _x[2]
    _jval[7] = 2 * _x[1]
    _jval[8] = -1
    _jval[9] = -1
    _jval[10] = 2 * _x[2]
    return INT(0)
  end

  function eval_gj_dense_c(n::INT, m::INT, J_ne::INT, x::Ptr{T}, g::Ptr{T},
                           jval::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _g = unsafe_wrap(Vector{T}, g, n)
    _jval = unsafe_wrap(Vector{T}, jval, J_ne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_expo{T}
    eval_gj_dense(_x, _g, _jval, _userdata)
  end

  eval_gj_dense_ptr = @eval @cfunction($eval_gj_dense_c, $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # compute the Hessian
  function eval_hl(x::Vector{T}, y::Vector{T}, hval::Vector{T}, userdata::userdata_expo{T})
    _hval[1] = 2 - 2 * (_y[2] + _userdata.p * _y[3] + _y[4])
    _hval[2] = 2 - 2 * (_y[2] + _y[3] + _y[5])
    return INT(0)
  end

  function eval_hl_c(n::INT, m::INT, H_ne::INT, x::Ptr{T}, y::Ptr{T},
                     hval::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _y = unsafe_wrap(Vector{T}, y, m)
    _hval = unsafe_wrap(Vector{T}, hval, H_ne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_expo{T}
    eval_hl(_x, _y, _hval, _userdata)
  end

  eval_hl_ptr = @eval @cfunction($eval_hl_c, $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # compute the dense Hessian
  function eval_hl_dense_c(x::Vector{T}, y::Vector{T}, hval::Vector{T}, userdata::userdata_expo{T})
    _hval[1] = 2 - 2 * (_y[2] + _userdata.p * _y[3] + _y[4])
    _hval[2] = 0
    _hval[3] = 2 - 2* (_y[2] + _y[3] + _y[5])
    return INT(0)
  end

  function eval_hl_dense_c(n::INT, m::INT, H_ne::INT, x::Ptr{T}, y::Ptr{T},
                           hval::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _y = unsafe_wrap(Vector{T}, y, m)
    _hval = unsafe_wrap(Vector{T}, hval, H_ne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_expo{T}
    eval_hl_dense(_x, _y, _hval, _userdata)
  end

  eval_hl_dense_ptr = @eval @cfunction($eval_hl_dense_c, $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{expo_control_type{T,INT}}()
  inform = Ref{expo_inform_type{T,INT}}()

  # Set user data
  userdata = userdata_expo{T}(9)
  userdata_ptr = pointer_from_objref(userdata)

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
      # @reset control[].print_level = INT(1)
      # @reset control[].tru_control.print_level = INT(1)
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
                                  userdata_ptr, status, n, m, j_ne_dense, h_ne_dense,
                                  c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                  eval_fc_ptr, eval_gj_dense_ptr, eval_hl_dense_ptr)
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
      @test test_expo(T, INT, mode="direct") == 0
    end
  end
end
