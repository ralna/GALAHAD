# test_arc.jl
# Simple code to test the Julia interface to ARC

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

# Custom userdata struct
mutable struct userdata_arc{T}
  p::T
end

function test_arc(::Type{T}, ::Type{INT}; mode::String="reverse", sls::String="sytr", dls::String="potr") where {T,INT}

  # Objective function
  function fun(x::Vector{T}, f::Vector{T}, userdata::userdata_arc{T})
    p = userdata.p
    f[1] = (x[1] + x[3] + p)^2 + (x[2] + x[3])^2 + cos(x[1])
    return INT(0)
  end

  function fun_c(n::INT, x::Ptr{T}, f::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _f = unsafe_wrap(Vector{T}, f, 1)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_arc{T}
    fun(_x, _f, _userdata)
  end

  fun_ptr = @eval @cfunction($fun_c, $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # Gradient of the objective
  function grad(x::Vector{T}, g::Vector{T}, userdata::userdata_arc{T})
    p = userdata.p
    g[1] = 2.0 * (x[1] + x[3] + p) - sin(x[1])
    g[2] = 2.0 * (x[2] + x[3])
    g[3] = 2.0 * (x[1] + x[3] + p) + 2.0 * (x[2] + x[3])
    return INT(0)
  end

  function grad_c(n::INT, x::Ptr{T}, g::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _g = unsafe_wrap(Vector{T}, g, n)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_arc{T}
    grad(_x, _g, _userdata)
  end

  grad_ptr = @eval @cfunction($grad_c, $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # Hessian of the objective
  function hess(x::Vector{T}, hval::Vector{T}, userdata::userdata_arc{T})
    hval[1] = 2.0 - cos(x[1])
    hval[2] = 2.0
    hval[3] = 2.0
    hval[4] = 2.0
    hval[5] = 4.0
    return INT(0)
  end

  function hess_c(n::INT, ne::INT, x::Ptr{T}, hval::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _hval = unsafe_wrap(Vector{T}, hval, ne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_arc{T}
    hess(_x, _hval, _userdata)
  end

  hess_ptr = @eval @cfunction($hess_c, $INT, ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # Dense Hessian
  function hess_dense(x::Vector{T}, hval::Vector{T}, userdata::userdata_arc{T})
    hval[1] = 2.0 - cos(x[1])
    hval[2] = 0.0
    hval[3] = 2.0
    hval[4] = 2.0
    hval[5] = 2.0
    hval[6] = 4.0
    return INT(0)
  end

  function hess_dense_c(n::INT, ne::INT, x::Ptr{T}, hval::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _hval = unsafe_wrap(Vector{T}, hval, ne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_arc{T}
    hess_dense(_x, _hval, _userdata)
  end

  hess_dense_ptr = @eval @cfunction($hess_dense_c, $INT, ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # Hessian-vector product
  function hessprod(x::Vector{T}, u::Vector{T}, v::Vector{T},
                    got_h::Bool, userdata::userdata_arc{T})
    u[1] = u[1] + 2.0 * (v[1] + v[3]) - cos(x[1]) * v[1]
    u[2] = u[2] + 2.0 * (v[2] + v[3])
    u[3] = u[3] + 2.0 * (v[1] + v[2] + 2.0 * v[3])
    return INT(0)
  end

  function hessprod_c(n::INT, x::Ptr{T}, u::Ptr{T}, v::Ptr{T},
                      got_h::Bool, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _u = unsafe_wrap(Vector{T}, u, n)
    _v = unsafe_wrap(Vector{T}, v, n)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_arc{T}
    hessprod(_x, _u, _v, got_h, _userdata)
  end

  hessprod_ptr = @eval @cfunction($hessprod_c, $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Bool, Ptr{Cvoid}))

  # Apply preconditioner
  function prec(x::Vector{T}, u::Vector{T}, v::Vector{T}, userdata::userdata_arc{T})
    u[1] = 0.5 * v[1]
    u[2] = 0.5 * v[2]
    u[3] = 0.25 * v[3]
    return INT(0)
  end

  function prec_c(n::INT, x::Ptr{T}, u::Ptr{T}, v::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _u = unsafe_wrap(Vector{T}, u, n)
    _v = unsafe_wrap(Vector{T}, v, n)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_arc{T}
    prec(_x, _u, _v, _userdata)
  end

  prec_ptr = @eval @cfunction($prec_c, $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # Objective function
  function fun_diag(x::Vector{T}, f::Vector{T}, userdata::userdata_arc{T})
    p = userdata.p
    f[1] = (x[3] + p)^2 + x[2]^2 + cos(x[1])
    return INT(0)
  end

  function fun_diag_c(n::INT, x::Ptr{T}, f::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _f = unsafe_wrap(Vector{T}, f, 1)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_arc{T}
    fun_diag(_x, _f, _userdata)
  end

  fun_diag_ptr = @eval @cfunction($fun_diag_c, $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # Gradient of the objective
  function grad_diag(x::Vector{T}, g::Vector{T}, userdata::userdata_arc{T})
    p = userdata.p
    g[1] = -sin(x[1])
    g[2] = 2.0 * x[2]
    g[3] = 2.0 * (x[3] + p)
    return INT(0)
  end

  function grad_diag_c(n::INT, x::Ptr{T}, g::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _g = unsafe_wrap(Vector{T}, g, n)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_arc{T}
    grad_diag(_x, _g, _userdata)
  end

  grad_diag_ptr = @eval @cfunction($grad_diag_c, $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # Hessian of the objective
  function hess_diag(x::Vector{T}, hval::Vector{T}, userdata::userdata_arc{T})
    hval[1] = -cos(x[1])
    hval[2] = 2.0
    hval[3] = 2.0
    return INT(0)
  end

  function hess_diag_c(n::INT, ne::INT, x::Ptr{T}, hval::Ptr{T}, userdata::Ptr{Cvoid})
    _x = unsafe_wrap(Vector{T}, x, n)
    _hval = unsafe_wrap(Vector{T}, hval, ne)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_arc{T}
    hess_diag(_x, _hval, _userdata)
  end

  hess_diag_ptr = @eval @cfunction($hess_diag_c, $INT, ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{arc_control_type{T,INT}}()
  inform = Ref{arc_inform_type{T,INT}}()

  # Set user data
  userdata = userdata_arc{T}(4)
  userdata_ptr = pointer_from_objref(userdata)

  # Set problem data
  n = INT(3)  # dimension
  ne = INT(5)  # Hesssian elements
  ne_dense = div(n * (n + 1), 2)  # dense Hesssian elements
  H_row = INT[1, 2, 3, 3, 3]  # Hessian H
  H_col = INT[1, 2, 1, 2, 3]  # NB lower triangle
  H_ptr = INT[1, 2, 3, 6]  # row pointers

  # Set storage
  g = zeros(T, n) # gradient
  st = ' '
  status = Ref{INT}()

  @printf(" Fortran sparse matrix indexing\n\n")

  if mode == "direct"
    @printf(" tests options for all-in-one storage format\n\n")

    for d in 1:5
      # Initialize ARC
      arc_initialize(T, INT, data, control, status)

      # Set user-defined control options
      # @reset control[].print_level = INT(1)

      # Linear solvers
      @reset control[].rqs_control.symmetric_linear_solver = galahad_linear_solver(sls)
      @reset control[].rqs_control.definite_linear_solver = galahad_linear_solver(dls)
      @reset control[].psls_control.definite_linear_solver = galahad_linear_solver(dls)
      @reset control[].dps_control.symmetric_linear_solver = galahad_linear_solver(sls)

      # Start from 1.5
      x = T[1.5, 1.5, 1.5]

      # sparse co-ordinate storage
      if d == 1
        st = 'C'
        arc_import(T, INT, control, data, status, n, "coordinate", ne, H_row, H_col, C_NULL)
        arc_solve_with_mat(T, INT, data, userdata_ptr, status, n, x, g, ne, fun_ptr, grad_ptr, hess_ptr, prec_ptr)
      end

      # sparse by rows
      if d == 2
        st = 'R'
        arc_import(T, INT, control, data, status, n, "sparse_by_rows", ne, C_NULL, H_col, H_ptr)
        arc_solve_with_mat(T, INT, data, userdata_ptr, status, n, x, g, ne, fun_ptr, grad_ptr, hess_ptr, prec_ptr)
      end

      # dense
      if d == 3
        st = 'D'
        arc_import(T, INT, control, data, status, n, "dense", ne_dense, C_NULL, C_NULL, C_NULL)
        arc_solve_with_mat(T, INT, data, userdata_ptr, status, n, x, g, ne_dense, fun_ptr, grad_ptr, hess_dense_ptr, prec_ptr)
      end

      # diagonal
      if d == 4
        st = 'I'
        arc_import(T, INT, control, data, status, n, "diagonal", n, C_NULL, C_NULL, C_NULL)
        arc_solve_with_mat(T, INT, data, userdata_ptr, status, n, x, g, n, fun_diag_ptr, grad_diag_ptr, hess_diag_ptr, prec_ptr)
      end

      # access by products
      if d == 5
        st = 'P'
        arc_import(T, INT, control, data, status, n, "absent", ne, C_NULL, C_NULL, C_NULL)
        arc_solve_without_mat(T, INT, data, userdata_ptr, status, n, x, g, fun_ptr, grad_ptr, hessprod_ptr, prec_ptr)
      end

      arc_information(T, INT, data, inform, status)

      if inform[].status[] == 0
        @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n", st,
                inform[].iter, inform[].obj, inform[].status)
      else
        @printf("%c: ARC_solve exit status = %1i\n", st, inform[].status)
      end

      # Delete internal workspace
      arc_terminate(T, INT, data, control, inform)
    end
  end

  if mode == "reverse"
    @printf(" tests reverse-communication options\n\n")

    # reverse-communication input/output
    eval_status = Ref{INT}()
    f = zeros(T, 1)
    u = zeros(T, n)
    v = zeros(T, n)
    H_val = zeros(T, ne)
    H_dense = zeros(T, div(n * (n + 1), 2))
    H_diag = zeros(T, n)

    for d in 1:5
      # Initialize ARC
      arc_initialize(T, INT, data, control, status)

      # Set user-defined control options
      # @reset control[].print_level = INT(1)

      # Linear solvers
      @reset control[].rqs_control.symmetric_linear_solver = galahad_linear_solver(sls)
      @reset control[].rqs_control.definite_linear_solver = galahad_linear_solver(dls)
      @reset control[].psls_control.definite_linear_solver = galahad_linear_solver(dls)
      @reset control[].dps_control.symmetric_linear_solver = galahad_linear_solver(sls)

      # Start from 1.5
      x = T[1.5, 1.5, 1.5]

      # sparse co-ordinate storage
      if d == 1
        st = 'C'
        arc_import(T, INT, control, data, status, n, "coordinate",
                   ne, H_row, H_col, C_NULL)

        terminated = false
        while !terminated # reverse-communication loop
          arc_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                     n, x, f[], g, ne, H_val, u, v)
          if status[] == 0 # successful termination
            terminated = true
          elseif status[] < 0 # error exit
            terminated = true
          elseif status[] == 2 # evaluate f
            eval_status[] = fun(x, f, userdata)
          elseif status[] == 3 # evaluate g
            eval_status[] = grad(x, g, userdata)
          elseif status[] == 4 # evaluate H
            eval_status[] = hess(x, H_val, userdata)
          elseif status[] == 6 # evaluate the product with P
            eval_status[] = prec(x, u, v, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      # sparse by rows
      if d == 2
        st = 'R'
        arc_import(T, INT, control, data, status, n, "sparse_by_rows",
                   ne, C_NULL, H_col, H_ptr)

        terminated = false
        while !terminated # reverse-communication loop
          arc_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                     n, x, f[], g, ne, H_val, u, v)
          if status[] == 0 # successful termination
            terminated = true
          elseif status[] < 0 # error exit
            terminated = true
          elseif status[] == 2 # evaluate f
            eval_status[] = fun(x, f, userdata)
          elseif status[] == 3 # evaluate g
            eval_status[] = grad(x, g, userdata)
          elseif status[] == 4 # evaluate H
            eval_status[] = hess(x, H_val, userdata)
          elseif status[] == 6 # evaluate the product with P
            eval_status[] = prec(x, u, v, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      # dense
      if d == 3
        st = 'D'
        arc_import(T, INT, control, data, status, n, "dense",
                   ne_dense, C_NULL, C_NULL, C_NULL)

        terminated = false
        while !terminated # reverse-communication loop
          arc_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                     n, x, f[], g, ne_dense,
                                     H_dense, u, v)
          if status[] == 0 # successful termination
            terminated = true
          elseif status[] < 0 # error exit
            terminated = true
          elseif status[] == 2 # evaluate f
            eval_status[] = fun(x, f, userdata)
          elseif status[] == 3 # evaluate g
            eval_status[] = grad(x, g, userdata)
          elseif status[] == 4 # evaluate H
            eval_status[] = hess_dense(x, H_dense, userdata)
          elseif status[] == 6 # evaluate the product with P
            eval_status[] = prec(x, u, v, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      # diagonal
      if d == 4
        st = 'I'
        arc_import(T, INT, control, data, status, n, "diagonal",
                   n, C_NULL, C_NULL, C_NULL)

        terminated = false
        while !terminated # reverse-communication loop
          arc_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                     n, x, f[], g, n, H_diag, u, v)
          if status[] == 0 # successful termination
            terminated = true
          elseif status[] < 0 # error exit
            terminated = true
          elseif status[] == 2 # evaluate f
            eval_status[] = fun_diag(x, f, userdata)
          elseif status[] == 3 # evaluate g
            eval_status[] = grad_diag(x, g, userdata)
          elseif status[] == 4 # evaluate H
            eval_status[] = hess_diag(x, H_diag, userdata)
          elseif status[] == 6 # evaluate the product with P
            eval_status[] = prec(x, u, v, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      # access by products
      if d == 5
        st = 'P'
        arc_import(T, INT, control, data, status, n, "absent",
                   ne, C_NULL, C_NULL, C_NULL)

        terminated = false
        while !terminated # reverse-communication loop
          arc_solve_reverse_without_mat(T, INT, data, status, eval_status,
                                        n, x, f[], g, u, v)
          if status[] == 0 # successful termination
            terminated = true
          elseif status[] < 0 # error exit
            terminated = true
          elseif status[] == 2 # evaluate f
            eval_status[] = fun(x, f, userdata)
          elseif status[] == 3 # evaluate g
            eval_status[] = grad(x, g, userdata)
          elseif status[] == 5 # evaluate H
            eval_status[] = hessprod(x, u, v, false, userdata)
          elseif status[] == 6 # evaluate the product with P
            eval_status[] = prec(x, u, v, userdata)
          else
            @printf(" the value %1i of status should not occur\n", status)
          end
        end
      end

      arc_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n", st,
                inform[].iter, inform[].obj, inform[].status)
      else
        @printf("%c: ARC_solve exit status = %1i\n", st, inform[].status)
      end

      # Delete internal workspace
      arc_terminate(T, INT, data, control, inform)
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
    @testset "ARC -- $T -- $INT" begin
      @testset "$mode communication" for mode in ("reverse", "direct")
        @test test_arc(T, INT; mode) == 0
      end
    end
  end
end
