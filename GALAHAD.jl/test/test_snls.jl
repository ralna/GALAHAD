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

Base.unsafe_convert(::Type{Ptr{Cvoid}}, userdata::userdata_snls) = pointer_from_objref(userdata)

function test_snls(::Type{T}, ::Type{INT}; mode::String="reverse", sls::String="sytr", dls::String="potr") where {T,INT}

  # ==================== define evaluation functions ====================

  # compute the residuals
  function res(x::Vector{T}, r::Vector{T}, userdata::userdata_snls{T})
    r[1] = x[1] * x[2] - userdata.p
    r[2] = x[2] * x[3] - one(T)
    r[3] = x[3] * x[4] - one(T)
    r[4] = x[4] * x[5] - one(T)
    return INT(0)
  end

  function res_c(n::INT, m_r::INT, x::Ptr{T}, r::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _r = unsafe_wrap(Vector{T}, r, m_r)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_snls{T}
    res(_x, _r, _userdata)
    return INT(0)
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
    return INT(0)
  end

  jac_ptr = @eval @cfunction($jac_c, $INT, 
                             ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # compute Jacobian-vector products
  function jacprod(x::Vector{T}, transpose::Bool, v::Vector{T}, p::Vector{T},
                   got_jr::Bool, userdata::userdata_snls{T})
    if transpose
      p[1] = x[2] * v[1]
      p[2] = x[3] * v[2] + x[1] * v[1]
      p[3] = x[4] * v[3] + x[2] * v[2]
      p[4] = x[5] * v[4] + x[3] * v[3]
      p[5] = x[4] * v[4]
    else
      p[1] = x[2] * v[1] + x[1] * v[2]
      p[2] = x[3] * v[2] + x[2] * v[3]
      p[3] = x[4] * v[3] + x[3] * v[4]
      p[4] = x[5] * v[4] + x[4] * v[5]
    end
    return INT(0)
  end

  function jacprod_c(n::INT, m_r::INT, x::Ptr{T}, transpose::Bool, 
                   v::Ptr{T}, p::Ptr{T}, got_jr::Bool, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _v = unsafe_wrap(Vector{T}, v, transpose ? m_r : n)
    _p = unsafe_wrap(Vector{T}, p, transpose ? n : m_r)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_snls{T}
    jacprod(_x, transpose, _v, _p, got_jr, _userdata)
    return INT(0)
  end

  jacprod_ptr = @eval @cfunction($jacprod_c, $INT, 
      ($INT, $INT, Ptr{$T}, Bool, Ptr{$T}, Ptr{$T}, Bool, Ptr{Cvoid}))

  # compute the index-th column of the Jacobian
  function jaccol(n::INT, x::Vector{T}, index::INT, 
                  val::Vector{T}, row::Vector{INT}, nz::Vector{INT}, 
                  got_jr::Bool, userdata::userdata_snls{T})
    if index == 1
      val[1] = x[2]
      row[1] = 1
      nz[1] = 1
    elseif index == n
      val[1] = x[n-1]
      row[1] = n
      nz[1] = 1
    else
      val[1] = x[index-1]
      row[1] = index
      val[2] = x[index+1]
      row[2] = index+1
      nz[1] = 2
    end
    return INT(0)
  end

  function jaccol_c(n::INT, m_r::INT, x::Ptr{T}, index::INT,
                      val::Ptr{T}, row::Ptr{INT}, nz::Ptr{INT},
                      got_jr::Bool, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _val = unsafe_wrap(Vector{T}, val, n)
    _row = unsafe_wrap(Vector{INT}, row, n)
    _nz = unsafe_wrap(Vector{INT}, nz, 1)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_snls{T}
    jaccol(n, _x, index, _val, _row, _nz, got_jr, _userdata)
    return INT(0)
  end

  jaccol_ptr = @eval @cfunction($jaccol_c, $INT, 
      ($INT, $INT, Ptr{$T}, $INT, Ptr{$T}, Ptr{$INT}, $Ptr{$INT},
       Bool, Ptr{Cvoid}))


  # compute a sparse product with the Jacobian
  function sjacprod(n::INT, m_r::INT, x::Vector{T}, transpose::Bool, 
                    v::Vector{T}, p::Vector{T}, free::Vector{INT}, n_free::INT, 
                    got_jr::Bool, userdata::userdata_snls{T})

    if transpose
      for i in 1:n_free
        j = free[i]
        if j == 1
          p[1] = x[2] * v[1]
        elseif j == n
          p[n] = x[m_r] * v[m_r]
        else
          p[j] = x[j-1] * v[j-1] + x[j+1] * v[j]
        end
      end
    else
      for i in 1:m_r
        p[i] = zero(T)
      end
      for i in 1:n_free
        j = free[i]
        val = v[j]
        if j == 1
          p[1] = p[1] + x[2] * val
        elseif j == n
          p[m_r] = p[m_r] + x[m_r] * val
        else
          p[j-1] = p[j-1] + x[j-1] * val
          p[j] = p[j] + x[j+1] * val
        end
      end
    end
    return INT(0)
  end

  function sjacprod_c(n::INT, m_r::INT, x::Ptr{T}, transpose::Bool, 
                      v::Ptr{T}, p::Ptr{T}, free::Ptr{INT}, n_free::INT,
                      got_jr::Bool, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _v = unsafe_wrap(Vector{T}, v, transpose ? m_r : n)
    _p = unsafe_wrap(Vector{T}, p, transpose ? n : m_r)
    _free = unsafe_wrap(Vector{INT}, free, n_free)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_snls{T}
    sjacprod(n, m_r, _x, transpose, _v, _p, _free, n_free, got_jr, _userdata)
    return INT(0)
  end

  sjacprod_ptr = @eval @cfunction($sjacprod_c, $INT, 
      ($INT, $INT, Ptr{$T}, Bool, Ptr{$T}, Ptr{$T}, Ptr{$INT}, $INT,
       Bool, Ptr{Cvoid}))

  # ==================== evaluation functions defined ===================

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{snls_control_type{T,INT}}()
  inform = Ref{snls_inform_type{T,INT}}()

  # Set user data
  userdata = userdata_snls{T}(4)

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
  x = Vector{T}(undef, n)
  y = Vector{T}(undef, m_c)
  z = Vector{T}(undef, n)
  r = Vector{T}(undef, m_r)
  g = Vector{T}(undef, n)
  x_stat = Vector{INT}(undef, n)

  status = Ref{INT}()

  @printf(" fortran sparse matrix indexing\n\n")

  if mode == "direct"
    for d in 1:2
      # Initialize SNLS
      snls_initialize(T, INT, data, control, inform)

      # Set user-defined control options
      # @reset control[].print_level = INT(1)
      @reset control[].stop_pg_absolute = T(0.00001)
      @reset control[].slls_control.sbls_control.definite_linear_solver =
        galahad_linear_solver(dls)
      @reset control[].slls_control.sbls_control.symmetric_linear_solver =
        galahad_linear_solver(sls)
      st = ""

      x = fill!(x, T(0.5))  # initial guess

      # solve when Jacobian is available via function calls
      if d == 1
        st = "JF"
        @reset control[].jacobian_available = INT(2)
        snls_import(T, INT, control, data, status, n, m_r, m_c,
                    "coordinate", jr_ne, Jr_row, Jr_col, INT(0), C_NULL, cohort)
        snls_solve_with_jac(T, INT, data, userdata, status, n, m_r, m_c,
                            x, y, z, r, g, x_stat, res_ptr, jr_ne, jac_ptr, w)
      end

      # solve when Jacobian products are available via function calls
      if d == 2
        st = "PF"
        @reset control[].jacobian_available = INT(1)
        snls_import_without_jac(T, INT, control, data, status, n, m_r, m_c, cohort)
        snls_solve_with_jacprod(T, INT, data, userdata, status, n, m_r, m_c,
                                x, y, z, r, g, x_stat,
                                res_ptr, jacprod_ptr, jaccol_ptr, sjacprod_ptr, w)
      end

      snls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf(" SNLS(%s):%6d iterations. Optimal objective value = %5.2f \
                status = %1d\n", st, inform[].iter, Float64(inform[].obj), inform[].status)
      else
        @printf(" SNLS(%s): exit status = %1d\n", st, inform[].status)
      end

      # Delete internal workspace
      snls_terminate(T, INT, data, control, inform)
    end
  end

  if mode == "reverse"
    # reverse-communication input/output
    Jr_val = Vector{T}(undef, jr_ne)
    mrn = max(m_r, n)
    eval_status = Ref{INT}()
    lvl = Ref{INT}(0)
    lvu = Ref{INT}(0)
    index = Ref{INT}(0)
    iv = Vector{INT}(undef, mrn)
    ip = Vector{INT}(undef, m_r)
    v = Vector{T}(undef, mrn)
    p = Vector{T}(undef, mrn)
    lp = zero(INT, 1)
    got_jr = false
    st = ""

    for d in 1:2
      # Initialize SNLS
      snls_initialize(T, INT, data, control, inform)

      # Set user-defined control options
      @reset control[].jacobian_available = INT(2) # Jacobian available
      # @reset control[].print_level = INT(1)
      @reset control[].stop_pg_absolute = T(0.00001)
      @reset control[].slls_control.sbls_control.definite_linear_solver =
        galahad_linear_solver(dls)
      @reset control[].slls_control.sbls_control.symmetric_linear_solver =
        galahad_linear_solver(sls)

      x = fill(T(0.5), n) # initial guess

      if d == 1
        # solve when Jacobian is available via reverse access
        st = "JR"
        @reset control[].jacobian_available = INT(2)
        snls_import(T, INT, control, data, status, n, m_r, m_c,
                    "coordinate", jr_ne, Jr_row, Jr_col, INT(0), C_NULL, cohort)
        terminated = false
        while !terminated # reverse-communication loop
          snls_solve_reverse_with_jac(T, INT, data, status, eval_status,
                                      n, m_r, m_c, x, y, z, r, g, x_stat,
                                      jr_ne, Jr_val, w)
          if status[] == 0 # successful termination
            terminated = true
          elseif status[] < 0 # error exit
            terminated = true
          elseif status[] == 2 # evaluate r
            eval_status[] = res(x, r, userdata)
          elseif status[] == 3 # evaluate Jr
            eval_status[] = jac(x, Jr_val, userdata)
          else
            @printf(" the value %i of status should not occur\n", status[])
          end
        end
      end

      if d == 2
        # solve when Jacobian products are available via reverse access
        st = "PR"
        @reset control[].jacobian_available = INT(1)
        snls_import_without_jac(T, INT, control, data, status, n, m_r, m_c, cohort)
        terminated = false
        while !terminated # reverse-communication loop
          snls_solve_reverse_with_jacprod(T, INT, data, status, eval_status,
                                          n, m_r, m_c, x, y, z, r, g, x_stat,
                                          v, iv, lvl, lvu, index, p, ip, lp[1], w)
          if status[] == 0 # successful termination
            terminated = true
          elseif status[] < 0 # error exit
            terminated = true
          elseif status[] == 2 # evaluate r
            eval_status[] = res(x, r, userdata)
            got_jr = false
          elseif status[] == 4 # evaluate p = Jr v
            eval_status[] = jacprod(x, false, v, p, got_jr, userdata)
          elseif status[] == 5 # evaluate p = Jr' v
            eval_status[] = jacprod(x, true, v, p, got_jr, userdata)
          elseif status[] == 6 # find the index-th column of Jr
            eval_status[] = jaccol(n, x, index[], p, ip, lp, got_jr, userdata)
          elseif status[] == 7 # evaluate p = J_o sparse(v)
            eval_status[] = sjacprod(n, m_r, x, false, v, p, iv, lvu[], got_jr,
                                     userdata)
          elseif status[] == 8 # evaluate p = sparse(Jr' v)
            eval_status[] = sjacprod(n, m_r, x, true, v, p, iv, lvu[], got_jr,
                                     userdata)
          else
            @printf(" the value %1d of status should not occur\n", status[])
          end
        end
      end

      snls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf(" SNLS(%s):%6d iterations. Optimal objective value = %5.2f \
        status = %1d\n", st, inform[].iter, Float64(inform[].obj), inform[].status)
      else
        @printf(" SNLS(%s): exit status = %1d\n", st, inform[].status)
      end

      # Delete internal workspace
      snls_terminate(T, INT, data, control, inform)
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
    @testset "SNLS -- $T -- $INT" begin
      @testset "$mode communication" for mode in ("reverse", "direct")
        @test test_snls(T, INT; mode) == 0
      end
    end
  end
end
