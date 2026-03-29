# test_snls.jl
# Simple code to test the Julia interface to SNLS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

# Custom userdata struct
mutable struct userdata_snls{T}
  p::T
end

function test_snls(::Type{T}, ::Type{INT}; sls::String="sytr", dls::String="potr") where {T,INT}

  # ==================== define evaluation functions ====================

  # compute the residuals
  function res(x::Vector{T}, r::Vector{T}, userdata::userdata_snls{T})
    println("x", x)
    p = userdata.p
    r[1] = x[1] * x[2] - p
    r[2] = x[2] * x[3] - 1.0
    r[3] = x[3] * x[4] - 1.0
    r[4] = x[4] * x[5] - 1.0
    println("r", r)
    return INT(0)
  end

  function res_c(n::INT, m_r::INT, x::Ptr{T}, r::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _r = unsafe_wrap(Vector{T}, r, m_r)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_snls{T}
    res(_x, _r, _userdata)
  end

  res_ptr = @eval @cfunction($res_c, $INT, 
                             ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # compute the Jacobian
  function jac(x::Vector{T}, jr_val::Vector{T}, userdata::userdata_snls{T})
    jr_val[1] = x[2]
    jr_val[2] = x[1]
    jr_val[3] = x[3]
    jr_val[4] = x[2]
    jr_val[5] = x[4]
    jr_val[6] = x[3]
    jr_val[7] = x[5]
    jr_val[8] = x[4]
    return INT(0)
  end

  function jac_c(n::INT, m_r::INT, jr_ne::INT, x::Ptr{T}, jr_val::Ptr{T}, 
                 userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _jr_val = unsafe_wrap(Vector{T}, jr_val, jr_ne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_snls{T}
    jac(_x, _jr_val, _userdata)
  end

  jac_ptr = @eval @cfunction($jac_c, $INT, 
                             ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # ==================== evaluation functions defined ===================

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{snls_control_type{T,INT}}()
  inform = Ref{snls_inform_type{T,INT}}()

  # Set user data
  userdata = userdata_snls{T}(1)
  userdata_ptr = pointer_from_objref(userdata)
  userdata.p = 4.0

  # Set problem dimensions
  n = INT(5)  # number of variables
  m_r = INT(4)  # number of residuals
  m_c = INT(2)  # number of cohorts

  # Set problem data
  jr_ne = INT(8) # Jacobian elements
  Jr_row = INT[1, 1, 2, 2, 3, 3, 4, 4]  # Jacobian Jr
  Jr_col = INT[1, 2, 2, 3, 3, 4, 4, 5]
  cohort = INT[1, 2, 0, 1, 2]  # cohorts
  w = T[1.0, 1.0, 1.0, 1.0]  # weights

  # Set space for output arrays
  x = Array{T, 1}(undef, n)  
  y = Array{T, 1}(undef, m_c)  
  z = Array{T, 1}(undef, n)  
  r = Array{T, 1}(undef, m_r)  
  g = Array{T, 1}(undef, n)  
  x_stat = Array{INT, 1}(undef, n)  

  status = Ref{INT}()

  @printf(" fortran sparse matrix indexing\n\n")

  # solve when Jacobian is available via function calls
  # ---------------------------------------------------

  # Initialize SNLS
  snls_initialize(T, INT, data, control, inform)

  # Set user-defined control options
  @reset control[].jacobian_available = INT(2) # Jacobian available
  @reset control[].f_indexing = true # fortran sparse matrix indexing
  @reset control[].print_level = INT(1)
  @reset control[].stop_pg_absolute = T(0.00001)
  @reset control[].slls_control.sbls_control.definite_linear_solver =
    galahad_linear_solver(dls)
  @reset control[].slls_control.sbls_control.symmetric_linear_solver =
    galahad_linear_solver(sls)

  #x = fill(0.5, n) # initial guess
  x = T[0.5, 0.5, 0.5, 0.5, 0.5]
  snls_import(T, INT, control, data, status, n, m_r, m_c,
              "coordinate", jr_ne, Jr_row, Jr_col, 0, C_NULL, cohort)
  snls_solve_with_jac(T, INT, data, userdata_ptr, status, n, m_r, m_c, 
                      x, y, z, r, g, x_stat, res_ptr, jr_ne, jac_ptr, w)
  snls_information(T, INT, data, inform, status)

  if inform[].status == 0
    @printf(" SNLS(JF):%6d iterations. Optimal objective value = %5.2f \
      status = %1d\n", inform[].iter, Float64(inform[].obj), inform[].status)
  else
    @printf(" SNLS(JF): exit status = %1d\n", inform[].status)
  end

  # Delete internal workspace
  snls_terminate(T, INT, data, control, inform)

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "SNLS -- $T -- $INT" begin
      @test test_snls(T, INT) == 0
    end
  end
end
