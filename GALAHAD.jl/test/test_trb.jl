# test_trb.jl
# Simple code to test the Julia interface to TRB

using GALAHAD
using Test
using Printf
using Accessors

# Custom userdata struct
struct userdata_trb
  p::Float64
end

function test_trb(::Type{T}) where T
  # Objective function
  function fun(n::Int, x::Vector{Float64}, f::Ref{Float64}, userdata::userdata_trb)
    p = userdata.p
    f[] = (x[1] + x[3] + p)^2 + (x[2] + x[3])^2 + cos(x[1])
    return 0
  end

  # Gradient of the objective
  function grad(n::Int, x::Vector{Float64}, g::Vector{Float64}, userdata::userdata_trb)
    p = userdata.p
    g[1] = 2.0 * (x[1] + x[3] + p) - sin(x[1])
    g[2] = 2.0 * (x[2] + x[3])
    g[3] = 2.0 * (x[1] + x[3] + p) + 2.0 * (x[2] + x[3])
    return 0
  end

  # Hessian of the objective
  function hess(n::Int, ne::Int, x::Vector{Float64}, hval::Vector{Float64},
                userdata::userdata_trb)
    hval[1] = 2.0 - cos(x[1])
    hval[2] = 2.0
    hval[3] = 2.0
    hval[4] = 2.0
    hval[5] = 4.0
    return 0
  end

  # Dense Hessian
  function hess_dense(n::Int, ne::Int, x::Vector{Float64}, hval::Vector{Float64},
                      userdata::userdata_trb)
    hval[1] = 2.0 - cos(x[1])
    hval[2] = 0.0
    hval[3] = 2.0
    hval[4] = 2.0
    hval[5] = 2.0
    hval[6] = 4.0
    return 0
  end

  # Hessian-vector product
  function hessprod(n::Int, x::Vector{Float64}, u::Vector{Float64}, v::Vector{Float64},
                    got_h::Bool, userdata::userdata_trb)
    u[1] = u[1] + 2.0 * (v[1] + v[3]) - cos(x[1]) * v[1]
    u[2] = u[2] + 2.0 * (v[2] + v[3])
    u[3] = u[3] + 2.0 * (v[1] + v[2] + 2.0 * v[3])
    return 0
  end

  # Sparse Hessian-vector product
  function shessprod(n::Int, x::Vector{Float64}, nnz_v::Cint, index_nz_v::Vector{Cint},
                     v::Vector{Float64}, nnz_u::Ref{Cint}, index_nz_u::Vector{Cint},
                     u::Vector{Float64}, got_h::Bool, userdata::userdata_trb)
    p = zeros(Float64, 3)
    used = falses(3)
    for i in 1:nnz_v
      j = index_nz_v[i]
      if j == 1
        p[1] = p[1] + 2.0 * v[1] - cos(x[1]) * v[1]
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
  function prec(n::Int, x::Vector{Float64}, u::Vector{Float64}, v::Vector{Float64},
                userdata::userdata_trb)
    u[1] = 0.5 * v[1]
    u[2] = 0.5 * v[2]
    u[3] = 0.25 * v[3]
    return 0
  end

  # Objective function
  function fun_diag(n::Int, x::Vector{Float64}, f::Ref{Float64}, userdata::userdata_trb)
    p = userdata.p
    f[] = (x[3] + p)^2 + x[2]^2 + cos(x[1])
    return 0
  end

  # Gradient of the objective
  function grad_diag(n::Int, x::Vector{Float64}, g::Vector{Float64},
                     userdata::userdata_trb)
    p = userdata.p
    g[1] = -sin(x[1])
    g[2] = 2.0 * x[2]
    g[3] = 2.0 * (x[3] + p)
    return 0
  end

  # Hessian of the objective
  function hess_diag(n::Int, ne::Int, x::Vector{Float64}, hval::Vector{Float64},
                     userdata::userdata_trb)
    hval[1] = -cos(x[1])
    hval[2] = 2.0
    hval[3] = 2.0
    return 0
  end

  # Hessian-vector product
  function hessprod_diag(n::Int, x::Vector{Float64}, u::Vector{Float64}, v::Vector{Float64},
                         got_h::Bool, userdata::userdata_trb)
    u[1] = u[1] + -cos(x[1]) * v[1]
    u[2] = u[2] + 2.0 * v[2]
    u[3] = u[3] + 2.0 * v[3]
    return 0
  end

  # Sparse Hessian-vector product
  function shessprod_diag(n::Int, x::Vector{Float64}, nnz_v::Cint,
                          index_nz_v::Vector{Cint}, v::Vector{Float64}, nnz_u::Ref{Cint},
                          index_nz_u::Vector{Cint}, u::Vector{Float64}, got_h::Bool,
                          userdata::userdata_trb)
    p = zeros(Float64, 3)
    used = falses(3)
    for i in 1:nnz_v
      j = index_nz_v[i]
      if j == 1
        p[1] = p[1] - cos(x[1]) * v[1]
        used[1] = true
      elseif j == 2
        p[2] = p[2] + 2.0 * v[2]
        used[2] = true
      elseif j == 3
        p[3] = p[3] + 2.0 * v[3]
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
  control = Ref{trb_control_type{Float64}}()
  inform = Ref{trb_inform_type{Float64}}()

  # Set user data
  userdata = userdata_trb(4.0)

  # Set problem data
  n = 3 # dimension
  ne = 5 # Hesssian elements
  x_l = Float64[-10, -10, -10]
  x_u = Float64[0.5, 0.5, 0.5]
  H_row = Cint[1, 2, 3, 3, 3]  # Hessian H
  H_col = Cint[1, 2, 1, 2, 3]  # NB lower triangle
  H_ptr = Cint[1, 2, 3, 6]  # row pointers

  # Set storage
  g = zeros(Float64, n) # gradient
  st = ' '
  status = Ref{Cint}()

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" tests reverse-communication options\n\n")

  # reverse-communication input/output
  eval_status = Ref{Cint}()
  nnz_v = Ref{Cint}()
  nnz_u = Ref{Cint}()
  f = Ref{Float64}(0.0)
  u = zeros(Float64, n)
  v = zeros(Float64, n)
  index_nz_u = zeros(Cint, n)
  index_nz_v = zeros(Cint, n)
  H_val = zeros(Float64, ne)
  H_dense = zeros(Float64, div(n * (n + 1), 2))
  H_diag = zeros(Float64, n)

  for d in 1:5
    # Initialize TRB
    trb_initialize(Float64, data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing
    # @reset control[].print_level = 1

    # Start from 1.5
    x = Float64[1.5, 1.5, 1.5]

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      trb_import(Float64, control, data, status, n, x_l, x_u, "coordinate", ne, H_row, H_col, C_NULL)

      terminated = false
      while !terminated # reverse-communication loop
        trb_solve_reverse_with_mat(Float64, data, status, eval_status, n, x, f[], g, ne, H_val, u, v)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate f
          eval_status[] = fun(n, x, f, userdata)
        elseif status[] == 3 # evaluate g
          eval_status[] = grad(n, x, g, userdata)
        elseif status[] == 4 # evaluate H
          eval_status[] = hess(n, ne, x, H_val, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status[] = prec(n, x, u, v, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # sparse by rows
    if d == 2
      st = 'R'
      trb_import(Float64, control, data, status, n, x_l, x_u, "sparse_by_rows", ne, C_NULL, H_col,
                 H_ptr)

      terminated = false
      while !terminated # reverse-communication loop
        trb_solve_reverse_with_mat(Float64, data, status, eval_status, n, x, f[], g, ne, H_val, u, v)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate f
          eval_status[] = fun(n, x, f, userdata)
        elseif status[] == 3 # evaluate g
          eval_status[] = grad(n, x, g, userdata)
        elseif status[] == 4 # evaluate H
          eval_status[] = hess(n, ne, x, H_val, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status[] = prec(n, x, u, v, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # dense
    if d == 3
      st = 'D'
      trb_import(Float64, control, data, status, n, x_l, x_u,
                 "dense", ne, C_NULL, C_NULL, C_NULL)

      terminated = false
      while !terminated # reverse-communication loop
        trb_solve_reverse_with_mat(Float64, data, status, eval_status, n, x, f[], g,
                                   div(n * (n + 1), 2), H_dense, u, v)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate f
          eval_status[] = fun(n, x, f, userdata)
        elseif status[] == 3 # evaluate g
          eval_status[] = grad(n, x, g, userdata)
        elseif status[] == 4 # evaluate H
          eval_status[] = hess_dense(n, div(n * (n + 1), 2), x, H_dense, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status[] = prec(n, x, u, v, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # diagonal
    if d == 4
      st = 'I'
      trb_import(Float64, control, data, status, n, x_l, x_u, "diagonal", ne, C_NULL, C_NULL, C_NULL)

      terminated = false
      while !terminated # reverse-communication loop
        trb_solve_reverse_with_mat(Float64, data, status, eval_status, n, x, f[], g, n, H_diag, u, v)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate f
          eval_status[] = fun_diag(n, x, f, userdata)
        elseif status[] == 3 # evaluate g
          eval_status[] = grad_diag(n, x, g, userdata)
        elseif status[] == 4 # evaluate H
          eval_status[] = hess_diag(n, n, x, H_diag, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status[] = prec(n, x, u, v, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # access by products
    if d == 5
      st = 'P'
      trb_import(Float64, control, data, status, n, x_l, x_u, "absent", ne, C_NULL, C_NULL, C_NULL)
      nnz_u = Ref{Cint}(0)

      terminated = false
      while !terminated # reverse-communication loop
        trb_solve_reverse_without_mat(Float64, data, status, eval_status, n, x, f[], g, u, v,
                                      index_nz_v, nnz_v, index_nz_u, nnz_u[])
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate f
          eval_status[] = fun(n, x, f, userdata)
        elseif status[] == 3 # evaluate g
          eval_status[] = grad(n, x, g, userdata)
        elseif status[] == 5 # evaluate H
          eval_status[] = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status[] = prec(n, x, u, v, userdata)
        elseif status[] == 7 # evaluate sparse Hessian-vect prod
          eval_status[] = shessprod(n, x, nnz_v[], index_nz_v, v, nnz_u, index_nz_u, u,
                                    false,
                                    userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # Record solution information
    trb_information(Float64, data, inform, status)

    # Print solution details
    if inform[].status[] == 0
      @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n", st,
              inform[].iter, inform[].obj, inform[].status)
    else
      @printf("%c: TRB_solve exit status = %1i\n", st, inform[].status)
    end

    # @printf("x: ")
    # for i = 1:n
    #   @printf("%f ", x[i])
    # end
    # @printf("\n")
    # @printf("gradient: ")
    # for i = 1:n
    #   @printf("%f ", g[i])
    # end
    # @printf("\n")

    # Delete internal workspace
    trb_terminate(Float64, data, control, inform)
  end
  return 0
end

@testset "TRB" begin
  @test test_trb(Float32) == 0
  @test test_trb(Float64) == 0
end
