# test_bgo.jl
# Simple code to test the Julia interface to BGO

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

# Custom userdata struct
struct userdata_bgo{T,INT}
  p::T
  freq::INT
  mag::INT
end

function test_bgo(::Type{T}, ::Type{INT}) where {T,INT}

  # Objective function
  function fun(x::Vector{T}, f::Ref{T}, userdata::userdata_bgo)
    p = userdata.p
    freq = userdata.freq
    mag = userdata.mag
    f[] = (x[1] + x[3] + p)^2 + (x[2] + x[3])^2 + mag * cos(freq * x[1]) + sum(x)
    return 0
  end

  # Gradient of the objective
  function grad(x::Vector{T}, g::Vector{T}, userdata::userdata_bgo)
    p = userdata.p
    freq = userdata.freq
    mag = userdata.mag
    g[1] = 2.0 * (x[1] + x[3] + p) - mag * freq * sin(freq * x[1]) + 1.0
    g[2] = 2.0 * (x[2] + x[3]) + 1.0
    g[3] = 2.0 * (x[1] + x[3] + p) + 2.0 * (x[2] + x[3]) + 1.0
    return 0
  end

  # Hessian of the objective
  function hess(x::Vector{T}, hval::Vector{T}, userdata::userdata_bgo)
    p = userdata.p
    freq = userdata.freq
    mag = userdata.mag
    hval[1] = 2.0 - mag * freq^2 * cos(freq * x[1])
    hval[2] = 2.0
    hval[3] = 2.0
    hval[4] = 4.0
    return 0
  end

  # Dense Hessian
  function hess_dense(x::Vector{T}, hval::Vector{T}, userdata::userdata_bgo)
    p = userdata.p
    freq = userdata.freq
    mag = userdata.mag
    hval[1] = 2.0 - mag * freq^2 * cos(freq * x[1])
    hval[2] = 0.0
    hval[3] = 2.0
    hval[4] = 2.0
    hval[5] = 4.0
    return 0
  end

  # Hessian-vector product
  function hessprod(x::Vector{T}, u::Vector{T}, v::Vector{T}, got_h::Bool, userdata::userdata_bgo)
    p = userdata.p
    freq = userdata.freq
    mag = userdata.mag
    u[1] = u[1] + 2.0 * (v[1] + v[3]) - mag * freq^2 * cos(freq * x[1]) * v[1]
    u[2] = u[2] + 2.0 * (v[2] + v[3])
    u[3] = u[3] + 2.0 * (v[1] + v[2] + 2.0 * v[3])
    return 0
  end

  # Sparse Hessian-vector product
  function shessprod(x::Vector{T}, nnz_v::INT, index_nz_v::Vector{INT},
                     v::Vector{T}, nnz_u::Ref{INT}, index_nz_u::Vector{INT},
                     u::Vector{T}, got_h::Bool, userdata::userdata_bgo)
    p = userdata.p
    freq = userdata.freq
    mag = userdata.mag
    p = zeros(T, 3)
    used = falses(3)
    for i in 1:nnz_v
      j = index_nz_v[i]
      if j == 1
        p[1] = p[1] + 2.0 * v[1] - mag * freq^2 * cos(freq * x[1]) * v[1]
        used[1] = true
        p[3] = p[3] + 2.0 * v[1]
        used[3] = true
      elseif j == 2
        p[2] = p[2] + 2.0 * v[2]
        used[2] = true
        p[3] = p[3] + 2.0 * v[2]
        used[3] = true
      elseif j == 3
        p[1] = p[1] + 2.0 * v[3]
        used[1] = true
        p[2] = p[2] + 2.0 * v[3]
        used[2] = true
        p[3] = p[3] + 4.0 * v[3]
        used[3] = true
      end
    end

    nnz_u[] = 0
    for j in 1:3
      if used[j]
        u[j] = p[j]
        nnz_u[] += 1
        index_nz_u[nnz_u[]] = j
      end
    end
    return 0
  end

  # Apply preconditioner
  function prec(x::Vector{T}, u::Vector{T}, v::Vector{T}, userdata::userdata_bgo)
    u[1] = 0.5 * v[1]
    u[2] = 0.5 * v[2]
    u[3] = 0.25 * v[3]
    return 0
  end

  # Objective function
  function fun_diag(x::Vector{T}, f::Ref{T}, userdata::userdata_bgo)
    p = userdata.p
    freq = userdata.freq
    mag = userdata.mag

    f[] = (x[3] + p)^2 + x[2]^2 + mag * cos(freq * x[1]) + x[1] + x[2] + x[3]
    return 0
  end

  # Gradient of the objective
  function grad_diag(x::Vector{T}, g::Vector{T}, userdata::userdata_bgo)
    p = userdata.p
    freq = userdata.freq
    mag = userdata.mag

    g[1] = -mag * freq * sin(freq * x[1]) + 1
    g[2] = 2.0 * x[2] + 1
    g[3] = 2.0 * (x[3] + p) + 1
    return 0
  end

  # Hessian of the objective
  function hess_diag(x::Vector{T}, hval::Vector{T}, userdata::userdata_bgo)
    freq = userdata.freq
    mag = userdata.mag

    hval[1] = -mag * freq^2 * cos(freq * x[1])
    hval[2] = 2.0
    hval[3] = 2.0
    return 0
  end

  # Hessian-vector product
  function hessprod_diag(x::Vector{T}, u::Vector{T}, v::Vector{T}, got_h::Bool, userdata::userdata_bgo)
    freq = userdata.freq
    mag = userdata.mag

    u[1] += -mag * freq^2 * cos(freq * x[1]) * v[1]
    u[2] += 2.0 * v[2]
    u[3] += 2.0 * v[3]
    return 0
  end

  # Sparse Hessian-vector product
  function shessprod_diag(x::Vector{T}, nnz_v::INT, index_nz_v::Vector{INT},
                          v::Vector{T}, nnz_u::Ref{INT}, index_nz_u::Vector{INT},
                          u::Vector{T}, got_h::Bool, userdata::userdata_bgo)
    freq = userdata.freq
    mag = userdata.mag

    p = zeros(3)
    used = falses(3)
    for i in 1:nnz_v
      j = index_nz_v[i]
      if j == 1
        p[1] -= mag * freq^2 * cos(freq * x[1]) * v[1]
        used[1] = true
      elseif j == 2
        p[2] += 2.0 * v[2]
        used[2] = true
      elseif j == 3
        p[3] += 2.0 * v[3]
        used[3] = true
      end
    end
    nnz_u[] = 0
    for j in 1:3
      if used[j]
        u[j] = p[j]
        nnz_u[] += 1
        index_nz_u[nnz_u[]] = j
      end
    end
    return 0
  end

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{bgo_control_type{T,INT}}()
  inform = Ref{bgo_inform_type{T,INT}}()

  # Set user data
  userdata = userdata_bgo(4.0, 10, 1000)

  # Set problem data
  n = INT(3) # dimension
  ne = INT(5) # Hesssian elements
  x_l = T[-10, -10, -10]
  x_u = T[0.5, 0.5, 0.5]
  H_row = INT[1, 2, 3, 3, 3]  # Hessian H
  H_col = INT[1, 2, 1, 2, 3]  # NB lower triangle
  H_ptr = INT[1, 2, 3, 6]  # row pointers

  # Set storage
  g = zeros(T, n) # gradient
  st = ' '
  status = Ref{INT}()

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" tests reverse-communication options\n\n")

  # reverse-communication input/output
  eval_status = Ref{INT}()
  nnz_u = Ref{INT}()
  nnz_v = Ref{INT}()
  f = Ref{T}(0.0)
  u = zeros(T, n)
  v = zeros(T, n)
  index_nz_u = zeros(INT, n)
  index_nz_v = zeros(INT, n)
  H_val = zeros(T, ne)
  H_dense = zeros(T, div(n * (n + 1), 2))
  H_diag = zeros(T, n)

  for d in 1:5

    # Initialize BGO
    bgo_initialize(T, INT, data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing
    @reset control[].attempts_max = INT(10000)
    @reset control[].max_evals = INT(20000)
    @reset control[].sampling_strategy = INT(3)
    @reset control[].trb_control.maxit = INT(100)
    # @reset control[].print_level = CINT(1)

    # Start from 0
    x = T[0.0, 0.0, 0.0]

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      bgo_import(T, INT, control, data, status, n, x_l, x_u,
                 "coordinate", ne, H_row, H_col, C_NULL)

      terminated = false
      while !terminated # reverse-communication loop
        bgo_solve_reverse_with_mat(T, INT, data, status, eval_status,
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
        elseif status[] == 5 # evaluate Hv product
          eval_status[] = hessprod(x, u, v, false, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status[] = prec(x, u, v, userdata)
        elseif status[] == 23 # evaluate f and g
          eval_status[] = fun(x, f, userdata)
          eval_status[] = grad(x, g, userdata)
        elseif status[] == 25 # evaluate f and Hv product
          eval_status[] = fun(x, f, userdata)
          eval_status[] = hessprod(x, u, v, false, userdata)
        elseif status[] == 35 # evaluate g and Hv product
          eval_status[] = grad(x, g, userdata)
          eval_status[] = hessprod(x, u, v, false, userdata)
        elseif status[] == 235 # evaluate f, g and Hv product
          eval_status[] = fun(x, f, userdata)
          eval_status[] = grad(x, g, userdata)
          eval_status[] = hessprod(x, u, v, false, userdata)
        else
          @printf(" the value %1i of status should not occur\n",
                  status)
        end
      end
    end

    # sparse by rows
    if d == 2
      st = 'R'
      bgo_import(T, INT, control, data, status, n, x_l, x_u,
                 "sparse_by_rows", ne, C_NULL, H_col, H_ptr)

      terminated = false
      while !terminated # reverse-communication loop
        bgo_solve_reverse_with_mat(T, INT, data, status, eval_status,
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
        elseif status[] == 5 # evaluate Hv product
          eval_status[] = hessprod(x, u, v, false, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status[] = prec(x, u, v, userdata)
        elseif status[] == 23 # evaluate f and g
          eval_status[] = fun(x, f, userdata)
          eval_status[] = grad(x, g, userdata)
        elseif status[] == 25 # evaluate f and Hv product
          eval_status[] = fun(x, f, userdata)
          eval_status[] = hessprod(x, u, v, false, userdata)
        elseif status[] == 35 # evaluate g and Hv product
          eval_status[] = grad(x, g, userdata)
          eval_status[] = hessprod(x, u, v, false, userdata)
        elseif status[] == 235 # evaluate f, g and Hv product
          eval_status[] = fun(x, f, userdata)
          eval_status[] = grad(x, g, userdata)
          eval_status[] = hessprod(x, u, v, false, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # dense
    if d == 3
      st = 'D'
      bgo_import(T, INT, control, data, status, n, x_l, x_u,
                 "dense", ne, C_NULL, C_NULL, C_NULL)

      terminated = false
      while !terminated # reverse-communication loop
        bgo_solve_reverse_with_mat(T, INT, data, status, eval_status,
                                   n, x, f[], g, div(n * (n + 1), 2),
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
        elseif status[] == 5 # evaluate Hv product
          eval_status[] = hessprod(x, u, v, false, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status[] = prec(x, u, v, userdata)
        elseif status[] == 23 # evaluate f and g
          eval_status[] = fun(x, f, userdata)
          eval_status[] = grad(x, g, userdata)
        elseif status[] == 25 # evaluate f and Hv product
          eval_status[] = fun(x, f, userdata)
          eval_status[] = hessprod(x, u, v, false, userdata)
        elseif status[] == 35 # evaluate g and Hv product
          eval_status[] = grad(x, g, userdata)
          eval_status[] = hessprod(x, u, v, false, userdata)
        elseif status[] == 235 # evaluate f, g and Hv product
          eval_status[] = fun(x, f, userdata)
          eval_status[] = grad(x, g, userdata)
          eval_status[] = hessprod(x, u, v, false, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # diagonal
    if d == 4
      st = 'I'
      bgo_import(T, INT, control, data, status, n, x_l, x_u,
                 "diagonal", ne, C_NULL, C_NULL, C_NULL)

      terminated = false
      while !terminated # reverse-communication loop
        bgo_solve_reverse_with_mat(T, INT, data, status, eval_status,
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
        elseif status[] == 5 # evaluate Hv product
          eval_status[] = hessprod_diag(x, u, v, false, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status[] = prec(x, u, v, userdata)
        elseif status[] == 23 # evaluate f and g
          eval_status[] = fun_diag(x, f, userdata)
          eval_status[] = grad_diag(x, g, userdata)
        elseif status[] == 25 # evaluate f and Hv product
          eval_status[] = fun_diag(x, f, userdata)
          eval_status[] = hessprod_diag(x, u, v, false, userdata)
        elseif status[] == 35 # evaluate g and Hv product
          eval_status[] = grad_diag(x, g, userdata)
          eval_status[] = hessprod_diag(x, u, v, false, userdata)
        elseif status[] == 235 # evaluate f, g and Hv product
          eval_status[] = fun_diag(x, f, userdata)
          eval_status[] = grad_diag(x, g, userdata)
          eval_status[] = hessprod_diag(x, u, v, false, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # access by products
    if d == 5
      st = 'P'
      bgo_import(T, INT, control, data, status, n, x_l, x_u,
                 "absent", ne, C_NULL, C_NULL, C_NULL)

      nnz_u = Ref{INT}(0)
      terminated = false
      while !terminated # reverse-communication loop
        bgo_solve_reverse_without_mat(T, INT, data, status, eval_status,
                                      n, x, f[], g, u, v, index_nz_v,
                                      nnz_v, index_nz_u, nnz_u[])
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate f
          eval_status[] = fun(x, f, userdata)
        elseif status[] == 3 # evaluate g
          eval_status[] = grad(x, g, userdata)
        elseif status[] == 5 # evaluate Hv product
          eval_status[] = hessprod(x, u, v, false, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status[] = prec(x, u, v, userdata)
        elseif status[] == 7 # evaluate sparse Hess-vect product
          eval_status[] = shessprod(x, nnz_v[], index_nz_v, v,
                                    nnz_u, index_nz_u, u,
                                    false, userdata)
        elseif status[] == 23 # evaluate f and g
          eval_status[] = fun(x, f, userdata)
          eval_status[] = grad(x, g, userdata)
        elseif status[] == 25 # evaluate f and Hv product
          eval_status[] = fun(x, f, userdata)
          eval_status[] = hessprod(x, u, v, false, userdata)
        elseif status[] == 35 # evaluate g and Hv product
          eval_status[] = grad(x, g, userdata)
          eval_status[] = hessprod(x, u, v, false, userdata)
        elseif status[] == 235 # evaluate f, g and Hv product
          eval_status[] = fun(x, f, userdata)
          eval_status[] = grad(x, g, userdata)
          eval_status[] = hessprod(x, u, v, false, userdata)
        else
          @printf(" the value %1i of status should not occur\n",
                  status)
        end
      end
    end

    # Record solution information
    bgo_information(T, INT, data, inform, status)

    if inform[].status == 0
      @printf("%c:%6i evaluations. Optimal objective value = %5.2f status = %1i\n", st,
              inform[].f_eval, inform[].obj, inform[].status)
    else
      @printf("%c: BGO_solve exit status = %1i\n", st, inform[].status)
    end

    # @printf("x: ")
    # for i in 1:n
    #   @printf("%f ", x[i])
    # end
    # @printf("\n")
    # @printf("gradient: ")
    # for i in 1:n
    #  @printf("%f ", g[i])
    # end
    # @printf("\n")

    # Delete internal workspace
    bgo_terminate(T, INT, data, control, inform)
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
    @testset "BGO -- $T -- $INT" begin
      @test test_bgo(T, INT) == 0
    end
  end
end
