# test_bnls.jl
# Simple code to test the Julia interface to BNLS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

# Custom userdata struct
mutable struct userdata_bnls{T,INT}
  p::T
  flag::INT
  flags::Vector{INT}
  eval_r::Function
  eval_jr::Function
  eval_jr_prod::Function
  eval_jr_prods::Function
  eval_jr_sprod::Function
end

function Base.unsafe_convert(::Type{Ptr{Cvoid}}, userdata::userdata_bnls)
  return pointer_from_objref(userdata)
end

function test_bnls(::Type{T}, ::Type{INT}; mode::String="reverse", sls::String="sytr", dls::String="potr") where {T,INT}

  # compute the residuals
  function res(x::Vector{T}, r::Vector{T}, userdata::userdata_bnls{T,INT})
    r[1] = x[1] * x[2] - userdata.p
    r[2] = x[2] * x[3] - one(T)
    r[3] = x[3] * x[4] - one(T)
    r[4] = x[4] * x[5] - one(T)
    return INT(0)
  end

  # compute the Jacobian
  function jac(x::Vector{T}, jr_val::Vector{T}, userdata::userdata_bnls{T,INT})
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

  # compute Jacobian-vector products
  function jacprod(x::Vector{T}, transpose::Bool, v::Vector{T}, p::Vector{T},
                   got_jr::Bool, userdata::userdata_bnls{T,INT})
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

  # compute a sparse product with the Jacobian
  function jacprods(n::INT, m_r::INT, x::Vector{T}, v::Vector{T}, p::Vector{T},
                    iv::Vector{INT}, lvl::INT, lvu::INT, ip::Vector{INT}, 
                    lp::Vector{INT}, got_jr::Bool, 
                    userdata::userdata_bnls{T,INT})
    if !isempty(ip) && !isempty(lp)
      userdata.flag = userdata.flag + 1
      lp[1] = 0
      for l = lvl:lvu
        j = iv[l]
        val = v[j]
        if j == 1
          i = 1
          if userdata.flags[i] < userdata.flag
            userdata.flags[i] = userdata.flag
            p[i] = x[i+1] * val
            lp[1] = lp[1] + 1
            ip[lp[1]] = i
          else
            p[i] = p[i] + x[i+1] * val
          end
        elseif j == n
          i = m_r
          if userdata.flags[i] < userdata.flag
            userdata.flags[i] = userdata.flag
            p[i] = x[i] * val
            lp[1] = lp[1] + 1
            ip[lp[1]] = i
          else
            p[i] = p[i] + x[i] * val
          end
        else
          i = j - 1
          if userdata.flags[i] < userdata.flag
            userdata.flags[i] = userdata.flag
            p[i] = x[i] * val
            lp[1] = lp[1] + 1
            ip[lp[1]] = i
          else
            p[i] = p[i] + x[i] * val
          end
          i = j
          if userdata.flags[i] < userdata.flag
            userdata.flags[i] = userdata.flag
            p[i] = x[i+1] * val
            lp[1] = lp[1] + 1
            ip[lp[1]] = i
          else
            p[i] = p[i] + x[i+1] * val
          end
        end
      end
    else
      for i = 1:m_r
        p[i] = zero(T)
      end
      for l = lvl:lvu
        j = iv[l]
        val = v[j]
        if j == 1
          i = 1
          p[i] = p[i] + x[i+1] * val
        elseif j == n
          i = m_r
          p[i] = p[i] + x[i] * val
        else
          i = j - 1
          p[i] = p[i] + x[i] * val
          i = j
          p[i] = p[i] + x[i+1] * val
        end
      end
    end
    return INT(0)
  end

  # compute a sparse product with the Jacobian or its transpose
  function sjacprod(n::INT, m_r::INT, x::Vector{T}, transpose::Bool,
                    v::Vector{T}, p::Vector{T}, free::Vector{INT}, n_free::INT,
                    got_jr::Bool, userdata::userdata_bnls{T,INT})
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

  # Callbacks
  callback_r = galahad_r(T, INT)
  callback_jr = galahad_jr(T, INT)
  callback_jr_prod = galahad_jr_prod(T, INT)
  callback_jr_sprod = galahad_jr_sprod(T, INT)
  callback_jr_prods = galahad_jr_prods(T, INT)

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{bnls_control_type{T,INT}}()
  inform = Ref{bnls_inform_type{T,INT}}()

  # Set problem data
  n = INT(5)  # variables
  m_r = INT(4)  # observations
  w = T[1.0, 1.0, 1.0, 1.0]  # weights
  jr_ne = INT(8)  # Jacobian elements
  Jr_row = INT[1, 1, 2, 2, 3, 3, 4, 4]  # Jacobian J
  Jr_col = INT[1, 2, 2, 3, 3, 4, 4, 5]
  Jr_val = zeros(T, jr_ne)

  # Set storage
  x_l = zeros(T, n)  # lower bounds
  x_u = zeros(T, n)  # upper bounds
  x = zeros(T, n)  # variables
  z = zeros(T, n)  # dual variables
  r = zeros(T, m_r)  # residual
  g = zeros(T, n)  # gradient
  x_stat = zeros(INT, n)  # variable status
  status = Ref{INT}()

  # set variable bounds
  for i in 1:n
    x_l[i] = zero(T)  # lower bound
    x_u[i] = one(T)   # upper bound
  end

  # Set user data
  p = T(4)
  flag = INT(0)  # current flag value
  flags = zeros(INT, m_r)  # array of flags
  userdata = userdata_bnls{T,INT}(p, flag, flags, res, jac, jacprod, jacprods, sjacprod)

  @printf(" fortran sparse matrix indexing\n\n")

  if mode == "direct"
    # solve via function calls
    for d in 1:2
    #for d in 1:0
      # Initialize BNLS
      bnls_initialize(T, INT, data, control, inform)

      # Set user-defined control options
      # @reset control[].out = INT(0)
      # @reset control[].blls_control.out = INT(0)
      #@reset control[].print_level = INT(10)
      #@reset control[].blls_control.print_level = INT(4)
      # @reset control[].bllsb_control.print_level = INT(1)
      # @reset control[].maxit = INT(10)
      # @reset control[].blls_control.maxit = INT(5)
      @reset control[].jacobian_available = INT(2)
      if T == Float32
        @reset control[].stop_pg_absolute = T(0.0001)
      else
        @reset control[].stop_pg_absolute = T(0.00001)
      end
      @reset control[].blls_control.sbls_control.definite_linear_solver =
        galahad_linear_solver(dls)
      @reset control[].blls_control.sbls_control.symmetric_linear_solver =
        galahad_linear_solver(sls)
      st = " "

      for i in 1:n
        x[i] = T(0.5)  # starting point
      end

      # solve when Jacobian is available via function calls
      # --------------------------------------------------- 
      if d == 1
        st = "JF"
        @reset control[].jacobian_available = INT(2)
        bnls_import(T, INT, control, data, status, n, m_r, "coordinate", jr_ne,
                    Jr_row, Jr_col, INT(0), C_NULL)
        bnls_solve_with_jac(T, INT, data, userdata, status, n, m_r, x_l, x_u, 
                            x, z, r, g, x_stat, callback_r, jr_ne, callback_jr, w)
      end

      # solve when Jacobian products are available via function calls
      # ------------------------------------------------------------- 
      if d == 2
        st = "PF"
        @reset control[].jacobian_available = INT(1)
        bnls_import_without_jac(T, INT, control, data, status, n, m_r)
        bnls_solve_with_jacprod(T, INT, data, userdata, status, n, m_r, 
                                x_l, x_u, x, z, r, g, x_stat, callback_r, 
                                callback_jr_prod, callback_jr_prods, callback_jr_sprod, w)
      end

      bnls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf(" BNLS(%s):%6d iterations. Optimal objective value = %5.2f status = %1d\n",
                st, inform[].iter, inform[].obj, inform[].status)
      else
        @printf(" BNLS(%s): exit status = %1d\n", st, inform[].status)
      end

      # Delete internal workspace
      bnls_terminate(T, INT, data, control, inform)
    end
  end

  if mode == "reverse"
    # reverse-communication input/output
    mnm = max(m_r, n)
    eval_status = Ref{INT}()
    lvl = Ref{INT}()
    lvu = Ref{INT}()
    iv = zeros(INT, mnm)
    #ip = zeros(INT, m_r)
    ip = zeros(INT, mnm)
    lp = zeros(INT, 1)
    v = zeros(T, mnm)
    p = zeros(T, mnm)
    got_jr = true

    # solve via reverse access
    # ------------------------ 
    #for d in 1:0
    for d in 1:2
      # Initialize BNLS
      bnls_initialize(T, INT, data, control, inform)

      # Set user-defined control options
      # @reset control[].print_level = INT(1)
      # @reset control[].maxit = INT(10)
      # @reset control[].blls_control.maxit = INT(5)
      if T == Float32
        @reset control[].stop_pg_absolute = T(0.0001)
      else
        @reset control[].stop_pg_absolute = T(0.00001)
      end
      @reset control[].blls_control.sbls_control.definite_linear_solver =
        galahad_linear_solver(dls)
      @reset control[].blls_control.sbls_control.symmetric_linear_solver =
        galahad_linear_solver(sls)
      st = " "

      for i in 1:n
        x[i] = T(0.5)  # starting point
      end

      if d == 1
        # solve when Jacobian is available via reverse access
        # ---------------------------------------------------
        st = "JR"
        @reset control[].jacobian_available = INT(2)
        bnls_import(T, INT, control, data, status, n, m_r, "coordinate", 
                    jr_ne, Jr_row, Jr_col, INT(0), C_NULL)

        terminated = false
        while !terminated # reverse-communication loop
          bnls_solve_reverse_with_jac(T, INT, data, status, eval_status, 
                                      n, m_r, x_l, x_u, x, z, r, g, x_stat, 
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
            @printf(" the value %1d of status should not occur\n", status[])
          end
        end
      end

      if d == 2
        # solve when Jacobian products are available via reverse access
        # -------------------------------------------------------------
        st = "PR"
        @reset control[].jacobian_available = INT(1)
        bnls_import_without_jac(T, INT, control, data, status, n, m_r)

        terminated = false
        while !terminated # reverse-communication loop
          bnls_solve_reverse_with_jacprod(T, INT, data, status, eval_status, 
                                          n, m_r, x_l, x_u, x, z, r, g, x_stat,
                                          v, iv, lvl, lvu, p, ip, lp[1], w)
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
          elseif status[] == 6 # evaluate p = Jr * sparse v
            eval_status[] = jacprods(n, m_r, x, v, p, iv, lvl[], lvu[], 
                                     INT[], INT[], got_jr, userdata)
          elseif status[] == 7 # evaluate p = sparse(Jr(x) * sparse v)
            eval_status[] = jacprods(n, m_r, x, v, p, iv, lvl[], lvu[], 
                                     ip, lp, got_jr, userdata)
          elseif status[] == 8 # evaluate p = sparse(Jr' v)
            eval_status[] = sjacprod(n, m_r, x, true, v, p, iv, lvu[], 
                                     got_jr, userdata)
          else
            @printf(" the value %1d of status should not occur\n", status[])
          end
        end
      end

      bnls_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf(" BNLS(%s):%6d iterations. Optimal objective value = %5.2f status = %1d\n",
                st, inform[].iter, inform[].obj, inform[].status)
      else
        @printf(" BNLS(%s): exit status = %1d\n", st, inform[].status)
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
      @testset "$mode communication" for mode in ("direct","reverse")
        @test test_bnls(T, INT; mode) == 0
      end
    end
  end
end
