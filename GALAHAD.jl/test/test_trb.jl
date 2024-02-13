# test_trb.jl
# Simple code to test the Julia interface to TRB

using GALAHAD
using Test
using Printf
using Accessors

# Custom userdata struct
struct userdata_type
  p::Float64
end

function test_trb()
  # Objective function
  function fun(n::Int, var::Vector{Float64}, f::Ref{Float64}, userdata::userdata_type)
    p = userdata.p
    f[] = (x[1] + x[3] + p)^2 + (x[2] + x[3])^2 + cos(x[1])
    return 0
  end

  # # Gradient of the objective
  # int grad(n::Int, var::Vector{Float64}, g::Vector{Float64}, userdata::userdata_type)
  # struct userdata_type *myuserdata = (struct userdata_type *) userdata
  # real_wp_ p = myuserdata->p

  # g[0] = 2.0 * (x[0] + x[2] + p) - sin(x[0])
  # g[1] = 2.0 * (x[1] + x[2])
  # g[2] = 2.0 * (x[0] + x[2] + p) + 2.0 * (x[1] + x[2])
  # return 0
  # ]

  # # Hessian of the objective
  # int hess(n::Int, n::Inte, var::Vector{Float64}, hval::Vector{Float64},
  #   userdata::userdata_type)
  # hval[0] = 2.0 - cos(x[0])
  # hval[1] = 2.0
  # hval[2] = 2.0
  # hval[3] = 2.0
  # hval[4] = 4.0
  # return 0
  # ]

  # # Dense Hessian
  # int hess_dense(n::Int, n::Inte, var::Vector{Float64}, hval::Vector{Float64},
  # userdata::userdata_type)
  # hval[0] = 2.0 - cos(x[0])
  # hval[1] = 0.0
  # hval[2] = 2.0
  # hval[3] = 2.0
  # hval[4] = 2.0
  # hval[5] = 4.0
  # return 0
  # ]

  # # Hessian-vector product
  # int hessprod(n::Int, var::Vector{Float64}, u::Vector{Float64}, var::Vector{Float64},
  #   bool got_h, userdata::userdata_type)
  # u[0] = u[0] + 2.0 * (v[0] + v[2]) - cos(x[0]) * v[0]
  # u[1] = u[1] + 2.0 * (v[1] + v[2])
  # u[2] = u[2] + 2.0 * (v[0] + v[1] + 2.0 * v[2])
  # return 0
  # ]

  # # Sparse Hessian-vector product
  # int shessprod(n::Int, var::Vector{Float64}, n::Intnz_v, const int index_nz_v[],
  #    var::Vector{Float64}, int *nnz_u, int index_nz_u[], u::Vector{Float64},
  #    bool got_h, userdata::userdata_type)
  # real_wp_ p[] = {0., 0., 0.]
  # bool used[] = {false, false, false]
  # for(int i = 0 i < nnz_v i++)
  # int j = index_nz_v[i]
  # switch(j)
  # case 1:
  # p[0] = p[0] + 2.0 * v[0] - cos(x[0]) * v[0]
  # used[0] = true
  # p[2] = p[2] + 2.0 * v[0]
  # used[2] = true
  # end
  # case 2:
  # p[1] = p[1] + 2.0 * v[1]
  # used[1] = true
  # p[2] = p[2] + 2.0 * v[1]
  # used[2] = true
  # end
  # case 3:
  # p[0] = p[0] + 2.0 * v[2]
  # used[0] = true
  # p[1] = p[1] + 2.0 * v[2]
  # used[1] = true
  # p[2] = p[2] + 4.0 * v[2]
  # used[2] = true
  # end
  # ]
  # ]
  # *nnz_u = 0
  # for(int j = 0 j < 3 j++)
  # if used[j])
  # u[j] = p[j]
  # *nnz_u = *nnz_u + 1
  # index_nz_u[*nnz_u-1] = j+1
  # ]
  # ]
  # return 0
  # ]

  # # Apply preconditioner
  # int prec(n::Int, var::Vector{Float64}, u::Vector{Float64}, var::Vector{Float64},
  #   userdata::userdata_type)
  #    u[0] = 0.5 * v[0]
  #    u[1] = 0.5 * v[1]
  #    u[2] = 0.25 * v[2]
  #    return 0
  # ]

  #  # Objective function
  # int fun_diag(n::Int, var::Vector{Float64}, real_wp_ *f, userdata::userdata_type)
  # struct userdata_type *myuserdata = (struct userdata_type *) userdata
  # real_wp_ p = myuserdata->p

  # *f = pow(x[2] + p, 2) + pow(x[1], 2) + cos(x[0])
  # return 0
  # ]

  # # Gradient of the objective
  # int grad_diag(n::Int, var::Vector{Float64}, g::Vector{Float64}, userdata::userdata_type)
  # struct userdata_type *myuserdata = (struct userdata_type *) userdata
  # real_wp_ p = myuserdata->p

  # g[0] = -sin(x[0])
  # g[1] = 2.0 * x[1]
  # g[2] = 2.0 * (x[2] + p)
  # return 0
  # ]

  # # Hessian of the objective
  # int hess_diag(n::Int, n::Inte, var::Vector{Float64}, hval::Vector{Float64},
  #    userdata::userdata_type)
  # hval[0] = -cos(x[0])
  # hval[1] = 2.0
  # hval[2] = 2.0
  # return 0
  # ]

  # # Hessian-vector product
  # int hessprod_diag(n::Int, var::Vector{Float64}, u::Vector{Float64}, var::Vector{Float64},
  #    bool got_h, userdata::userdata_type)
  # u[0] = u[0] + - cos(x[0]) * v[0]
  # u[1] = u[1] + 2.0 * v[1]
  # u[2] = u[2] + 2.0 * v[2]
  # return 0
  # ]

  # # Sparse Hessian-vector product
  # int shessprod_diag(n::Int, var::Vector{Float64}, n::Intnz_v,
  # const int index_nz_v[],
  # var::Vector{Float64}, int *nnz_u, int index_nz_u[],
  # u::Vector{Float64}, bool got_h, userdata::userdata_type)
  # real_wp_ p[] = {0., 0., 0.]
  # bool used[] = {false, false, false]
  # for(int i = 0 i < nnz_v i++)
  # int j = index_nz_v[i]
  # switch(j)
  # case 0:
  # p[0] = p[0] - cos(x[0]) * v[0]
  # used[0] = true
  # end
  # case 1:
  # p[1] = p[1] + 2.0 * v[1]
  # used[1] = true
  # end
  # case 2:
  # p[2] = p[2] + 2.0 * v[2]
  # used[2] = true
  # end
  # ]
  # ]
  # *nnz_u = 0
  # for(int j = 0 j < 3 j++)
  # if used[j])
  # u[j] = p[j]
  # *nnz_u = *nnz_u + 1
  # index_nz_u[*nnz_u-1] = j+1
  # ]
  # ]
  # return 0
  # ]
  # end

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{trb_control_type{Float64}}()
  inform = Ref{trb_inform_type{Float64}}()

  # Set user data
  userdata = userdata_type(4.0)

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
  @printf(" tests options for all-in-one storage format\n\n")

  for d in 1:5

    # Initialize TRB
    trb_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing
    # @reset control[].print_level = 1

    # Start from 1.5
    x = Float64[1.5, 1.5, 1.5]

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      trb_import(control, data, status, n, x_l, x_u,
                 "coordinate", ne, H_row, H_col, Cint[])

      trb_solve_with_mat(data, userdata, status, n, x, g, ne,
                         fun, grad, hess, prec)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      trb_import(control, data, status, n, x_l, x_u,
                 "sparse_by_rows", ne, Cint[], H_col, H_ptr)

      trb_solve_with_mat(data, userdata, status, n, x, g, ne,
                         fun, grad, hess, prec)
    end

    # dense
    if d == 3
      st = 'D'
      trb_import(control, data, status, n, x_l, x_u,
                 "dense", ne, Cint[], Cint[], Cint[])

      trb_solve_with_mat(data, userdata, status, n, x, g, ne,
                         fun, grad, hess_dense, prec)
    end

    # diagonal
    if d == 4
      st = 'I'
      trb_import(control, data, status, n, x_l, x_u,
                 "diagonal", ne, Cint[], Cint[], Cint[])

      trb_solve_with_mat(data, userdata, status, n, x, g, ne,
                         fun_diag, grad_diag, hess_diag, prec)
    end

    # access by products
    if d == 5
      st = 'P'
      trb_import(control, data, status, n, x_l, x_u,
                 "absent", ne, Cint[], Cint[], Cint[])

      trb_solve_without_mat(data, userdata, status, n, x, g,
                            fun, grad, hessprod, shessprod, prec)
    end

    # Record solution information
    trb_information(data, inform, status)

    # Print solution details
    if inform[].status == 0
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
    trb_terminate(data, control, inform)
  end

  @printf("\n tests reverse-communication options\n\n")

  # reverse-communication input/output
  eval_status = Ref{Cint}()
  nnz_v = Ref{Cint}()
  nnz_u = Ref{Cint}()
  f = 0.0
  u = zeros(Float64, n)
  v = zeros(Float64, n)
  index_nz_u = zeros(Cint, n)
  index_nz_v = zeros(Cint, n)
  H_val = zeros(Float64, ne)
  H_dense = zeros(Float64, div(n * (n + 1), 2))
  H_diag = zeros(Float64, n)

  for d in 1:5

    # Initialize TRB
    trb_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing
    # @reset control[].print_level = 1

    # Start from 1.5
    x = Float64[1.5, 1.5, 1.5]

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      trb_import(control, data, status, n, x_l, x_u, "coordinate", ne, H_row, H_col, Cint[])

      terminated = false
      while !terminated # reverse-communication loop
        trb_solve_reverse_with_mat(data, status, eval_status, n, x, f, g, ne, H_val, u, v)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate f
          eval_status = fun(n, x, f, userdata)
        elseif status[] == 3 # evaluate g
          eval_status = grad(n, x, g, userdata)
        elseif status[] == 4 # evaluate H
          eval_status = hess(n, ne, x, H_val, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status = prec(n, x, u, v, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # sparse by rows
    if d == 2
      st = 'R'
      trb_import(control, data, status, n, x_l, x_u, "sparse_by_rows", ne, Cint[], H_col,
                 H_ptr)

      terminated = false
      while !terminated # reverse-communication loop
        trb_solve_reverse_with_mat(data, status, eval_status, n, x, f, g, ne, H_val, u, v)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate f
          eval_status = fun(n, x, f, userdata)
        elseif status[] == 3 # evaluate g
          eval_status = grad(n, x, g, userdata)
        elseif status[] == 4 # evaluate H
          eval_status = hess(n, ne, x, H_val, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status = prec(n, x, u, v, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # dense
    if d == 3
      st = 'D'
      trb_import(control, data, status, n, x_l, x_u,
                 "dense", ne, Cint[], Cint[], Cint[])

      terminated = false
      while !terminated # reverse-communication loop
        trb_solve_reverse_with_mat(data, status, eval_status, n, x, f, g, n * (n + 1) / 2,
                                   H_dense, u, v)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate f
          eval_status = fun(n, x, f, userdata)
        elseif status[] == 3 # evaluate g
          eval_status = grad(n, x, g, userdata)
        elseif status[] == 4 # evaluate H
          eval_status = hess_dense(n, div(n * (n + 1), 2), x, H_dense, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status = prec(n, x, u, v, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # diagonal
    if d == 4
      st = 'I'
      trb_import(control, data, status, n, x_l, x_u, "diagonal", ne, Cint[], Cint[], Cint[])

      terminated = false
      while !terminated # reverse-communication loop
        trb_solve_reverse_with_mat(data, status, eval_status, n, x, f, g, n, H_diag, u, v)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate f
          eval_status = fun_diag(n, x, f, userdata)
        elseif status[] == 3 # evaluate g
          eval_status = grad_diag(n, x, g, userdata)
        elseif status[] == 4 # evaluate H
          eval_status = hess_diag(n, n, x, H_diag, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status = prec(n, x, u, v, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # access by products
    if d == 5
      st = 'P'
      trb_import(control, data, status, n, x_l, x_u, "absent", ne, Cint[], Cint[], Cint[])
      nnz_u = 0

      terminated = false
      while !terminated # reverse-communication loop
        trb_solve_reverse_without_mat(data, status, eval_status, n, x, f, g, u, v,
                                      index_nz_v, nnz_v, index_nz_u, nnz_u)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate f
          eval_status = fun(n, x, f, userdata)
        elseif status[] == 3 # evaluate g
          eval_status = grad(n, x, g, userdata)
        elseif status[] == 5 # evaluate H
          eval_status = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status = prec(n, x, u, v, userdata)
        elseif status[] == 7 # evaluate sparse Hessian-vect prod
          eval_status = shessprod(n, x, nnz_v, index_nz_v, v, nnz_u, index_nz_u, u, false,
                                  userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # Record solution information
    trb_information(data, inform, status)

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
    trb_terminate(data, control, inform)
  end
  return 0
end

@testset "TRB" begin
  @test test_trb() == 0
end
