# test_expo.jl
# Simple code to test the Julia interface to EXPO

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

# Custom userdata struct
struct userdata_expo{T}
  p::T
end

function test_expo(::Type{T}, ::Type{INT}; sls::String="sytr", dls::String="potr") where {T,INT}

  # compute the objective and constraints
  function eval_fc(x::Vector{T}, f::T, c::Vector{T}, userdata::userdata_expo)
    f = x[1]^2 + x[2]^2
    c[1] = x[1] + x[2] - 1.0
    c[2] = x[1]^2 + x[2]^2 - 1.0
    c[3] = userdata.p * x[1]^2 + x[2]^2 - userdata.p
    c[4] = x[1]^2 - x[2]
    c[5] = x[2]^2 - x[1]
    return 0
  end

  # compute the gradient and Jacobian
  function eval_gj(x::Vector{T}, g::Vector{T}, jval::Vector{T}, 
                   userdata::userdata_expo)
    g[1] = 2.0 * x[1]
    g[2] = 2.0 * x[2]
    jval[1] = 1.0
    jval[2] = 1.0
    jval[3] = 2.0 * x[1]
    jval[4] = 2.0 * x[2]
    jval[5] = 2.0 * userdata.p * x[1]
    jval[6] = 2.0 * x[2]
    jval[7] = 2.0 * x[1]
    jval[8] = -1.0
    Jval[9] = -1.0
    jval[10] = 2.0 * x[2]
    return 0
  end

  # compute the Hessian
  function eval_hl(x::Vector{T}, y::Vector{T}, hval::Vector{T}, 
                   userdata::userdata_expo)
    hval[1] = 2.0 - 2.0 * (y[2] + userdata.p * y[3] + y[4])
    hval[2] = 2.0 - 2.0 * (y[2] + y[3] + y[5])
    return 0
  end

  # compute the dense Hessian
  function eval_hl_dense(x::Vector{T}, y::Vector{T}, hval::Vector{T}, 
                         userdata::userdata_expo)
    hval[1] = 2.0 - 2.0 * (y[2] + userdata.p * y[3] + y[4])
    hval[2] = 0.0
    hval[3] = 2.0 - 2.0 * (y[2] + y[3] + y[5])
    return 0
  end

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{expo_control_type{T,INT}}()
  inform = Ref{expo_inform_type{T,INT}}()

  # Set user data
  userdata = userdata_expo(1.0)

  # Set problem data
  n = INT(2)  # variables
  m = INT(5)  # constraints
  j_ne = INT(10) # Jacobian elements
  h_ne = INT(2)  # Hesssian elements
  J_row = INT[1, 2, 2, 3, 3]  # Jacobian J
  J_col = INT[1, 1, 2, 1, 2]  #
  J_ptr = INT[1, 2, 4, 6]  # row pointers
  H_row = INT[1, 2]  # Hessian H
  H_col = INT[1, 2]  # NB lower triangle
  H_ptr = INT[1, 2, 3]  # row pointers
  c_l = T[0.0, 0.0, 0.0, 0.0, 0.0]  # constraint lower bound
  c_u = T[Inf, Inf, Inf, Inf, Inf]  # constraint upper bound
  x_l = T[-50.0, -50.0]  # variable lower bound
  x_u = T[50.0, 50.0]  # variable upper bound

  # Set storage
  y = zeros(T, m) # multipliers
  z = zeros(T, m) # dual variables
  c = zeros(T, m) # constraints
  gl = zeros(T, n) # gradient
  st = ' '
  status = Ref{INT}()
  @reset userdata[].p = T(9.0)

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" test direct-communication options\n\n")

  for d in 1:4
    # Initialize EXPO
    expo_initialize(T, INT, data, control, inform)

    # Linear solvers
    @reset control[].ssls_control.definite_linear_solver = galahad_linear_solver(dls)
    @reset control[].tru_control.symmetric_linear_solver = galahad_linear_solver(sls)
    @reset control[].tru_control.definite_linear_solver = galahad_linear_solver(dls)

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
                                userdata, status, n, m, j_ne, h_ne,
                                c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                eval_fc, eval_gj, eval_hl)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      expo_import(T, INT, control, data, status, n, m,
                  "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                  "sparse_by_rows", h_ne, C_NULL, H_col, H_ptr )

      expo_solve_hessian_direct(T, INT, data, 
                                userdata, status, n, m, j_ne, h_ne,
                                c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                eval_fc, eval_gj, eval_hl)
    end

    # dense
    if d == 3
      st = 'D'
      expo_import(T, INT, control, data, status, n, m,
                  "dense", j_ne, C_NULL, C_NULL, C_NULL,
                  "dense", h_ne, C_NULL, C_NULL, C_NULL )

      expo_solve_hessian_direct(T, INT, data, 
                                userdata, status, n, m, j_ne, h_ne,
                                c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                eval_fc, eval_gj, eval_hl_dense)
    end

    # diagonal
    if d == 4
      st = 'I'
      expo_import(T, INT, control, data, status, n, m,
                  "sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
                  "diagonal", h_ne, C_NULL, C_NULL, C_NULL )

      expo_solve_hessian_direct(T, INT, data, 
                                userdata, status, n, m, j_ne, h_ne,
                                c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                eval_fc, eval_gj, eval_hl)
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
      @test test_expo(T, INT) == 0
    end
  end
end
