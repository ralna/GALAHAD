# test_dgo.jl
# Simple code to test the Julia interface to DGO

using GALAHAD
using Test
using Printf
using Accessors

# Custom userdata struct
struct userdata_type
  p::Float64
  freq::Float64
  mag::Float64
end

function test_dgo()

  # Objective function
  function fun(n::Int, var::Vector{Float64}, f::Ref{Float64}, userdata::userdata_type)
    p = userdata.p
    freq = userdata.freq
    mag = userdata.mag

    f[] = (x[1] + x[3] + p)^2 + (x[2] + x[3])^2 + mag * cos(freq * x[1]) + x[1] + x[2] +
          x[3]
    return 0
  end

  # Gradient of the objective
  function grad(n::Int, var::Vector{Float64}, g::Vector{Float64}, userdata::userdata_type)
    p = userdata.p
    freq = userdata.freq
    mag = userdata.mag

    g[1] = 2.0 * (x[1] + x[3] + p) - mag * freq * sin(freq * x[1]) + 1
    g[2] = 2.0 * (x[2] + x[3]) + 1
    g[3] = 2.0 * (x[1] + x[3] + p) + 2.0 * (x[2] + x[3]) + 1
    return 0
  end

  # # Hessian of the objective
  # int hess(int n,
  #   int ne,
  #   var::Vector{Float64},
  #   hval::Vector{Float64},
  #   const void *userdata)
  # struct userdata_type *myuserdata = (struct userdata_type *) userdata
  # real_wp_ freq = myuserdata->freq
  # real_wp_ mag = myuserdata->mag

  # hval[0] = 2.0 - mag * freq * freq * cos(freq*x[0])
  # hval[1] = 2.0
  # hval[2] = 2.0
  # hval[3] = 2.0
  # hval[4] = 4.0
  # return 0
  # ]

  # # Dense Hessian
  # int hess_dense(int n,
  # int ne,
  # var::Vector{Float64},
  # hval::Vector{Float64},
  # const void *userdata)
  # struct userdata_type *myuserdata = (struct userdata_type *) userdata
  # real_wp_ freq = myuserdata->freq
  # real_wp_ mag = myuserdata->mag

  # hval[0] = 2.0 - mag * freq * freq * cos(freq*x[0])
  # hval[1] = 0.0
  # hval[2] = 2.0
  # hval[3] = 2.0
  # hval[4] = 2.0
  # hval[5] = 4.0
  # return 0
  # ]

  # # Hessian-vector product
  # int hessprod(int n,
  #   var::Vector{Float64},
  #   u::Vector{Float64},
  #   var::Vector{Float64},
  #   bool got_h,
  #   const void *userdata)
  # struct userdata_type *myuserdata = (struct userdata_type *) userdata
  # real_wp_ freq = myuserdata->freq
  # real_wp_ mag = myuserdata->mag

  # u[0] = u[0] + 2.0 * (v[0] + v[2])
  #    - mag * freq * freq * cos(freq*x[0]) * v[0]
  # u[1] = u[1] + 2.0 * (v[1] + v[2])
  # u[2] = u[2] + 2.0 * (v[0] + v[1] + 2.0 * v[2])
  # return 0
  # ]

  # # Sparse Hessian-vector product
  # int shessprod(int n,
  #    var::Vector{Float64},
  #    int nnz_v,
  #    const int index_nz_v[],
  #    var::Vector{Float64},
  #    int *nnz_u,
  #    int index_nz_u[],
  #    u::Vector{Float64},
  #    bool got_h,
  #    const void *userdata)
  # struct userdata_type *myuserdata = (struct userdata_type *) userdata
  # real_wp_ freq = myuserdata->freq
  # real_wp_ mag = myuserdata->mag

  # real_wp_ p[] = {0., 0., 0.]
  # bool used[] = {false, false, false]
  # for(int i = 0 i < nnz_v i++)
  # int j = index_nz_v[i]
  # switch(j)
  # case 1:
  # p[0] = p[0] + 2.0 * v[0]
  #  - mag * freq * freq * cos(freq*x[0]) * v[0]
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
  # int prec(int n,
  #   var::Vector{Float64},
  #   u::Vector{Float64},
  #   var::Vector{Float64},
  #   const void *userdata)
  #    u[0] = 0.5 * v[0]
  #    u[1] = 0.5 * v[1]
  #    u[2] = 0.25 * v[2]
  #    return 0
  # ]

  # # Objective function
  # int fun_diag(int n,
  #   var::Vector{Float64},
  #   real_wp_ *f,
  #   const void *userdata)
  # struct userdata_type *myuserdata = (struct userdata_type *) userdata
  # real_wp_ p = myuserdata->p
  # real_wp_ freq = myuserdata->freq
  # real_wp_ mag = myuserdata->mag

  # *f = pow(x[2] + p, 2) + pow(x[1], 2) + mag * cos(freq*x[0])
  #  + x[0] + x[1] + x[2]
  # return 0
  # ]

  # # Gradient of the objective
  # int grad_diag(int n,
  #    var::Vector{Float64},
  #    g::Vector{Float64},
  #    const void *userdata)
  # struct userdata_type *myuserdata = (struct userdata_type *) userdata
  # real_wp_ p = myuserdata->p
  # real_wp_ freq = myuserdata->freq
  # real_wp_ mag = myuserdata->mag

  # g[0] = -mag * freq * sin(freq*x[0]) + 1
  # g[1] = 2.0 * x[1] + 1
  # g[2] = 2.0 * (x[2] + p) + 1
  # return 0
  # ]

  # # Hessian of the objective
  # int hess_diag(int n,
  #    int ne,
  #    var::Vector{Float64},
  #    hval::Vector{Float64},
  #    const void *userdata)
  # struct userdata_type *myuserdata = (struct userdata_type *) userdata
  # real_wp_ freq = myuserdata->freq
  # real_wp_ mag = myuserdata->mag

  # hval[0] = -mag * freq * freq * cos(freq*x[0])
  # hval[1] = 2.0
  # hval[2] = 2.0
  # return 0
  # ]

  # # Hessian-vector product
  # int hessprod_diag(int n,
  #    var::Vector{Float64},
  #    u::Vector{Float64},
  #    var::Vector{Float64},
  #    bool got_h,
  #    const void *userdata)
  # struct userdata_type *myuserdata = (struct userdata_type *) userdata
  # real_wp_ freq = myuserdata->freq
  # real_wp_ mag = myuserdata->mag

  # u[0] = u[0] + -mag * freq * freq * cos(freq*x[0]) * v[0]
  # u[1] = u[1] + 2.0 * v[1]
  # u[2] = u[2] + 2.0 * v[2]
  # return 0
  # ]

  # # Sparse Hessian-vector product
  # int shessprod_diag(int n,
  # var::Vector{Float64},
  # int nnz_v,
  # const int index_nz_v[],
  # var::Vector{Float64},
  # int *nnz_u,
  # int index_nz_u[],
  # u::Vector{Float64},
  # bool got_h,
  # const void *userdata)
  # struct userdata_type *myuserdata = (struct userdata_type *) userdata
  # real_wp_ freq = myuserdata->freq
  # real_wp_ mag = myuserdata->mag

  # real_wp_ p[] = {0., 0., 0.]
  # bool used[] = {false, false, false]
  # for(int i = 0 i < nnz_v i++)
  # int j = index_nz_v[i]
  # switch(j)
  # case 1:
  # p[0] = p[0] - mag * freq * freq * cos(freq*x[0]) * v[0]
  # used[0] = true
  # end
  # case 2:
  # p[1] = p[1] + 2.0 * v[1]
  # used[1] = true
  # end
  # case 3:
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
  control = Ref{dgo_control_type{Float64}}()
  inform = Ref{dgo_inform_type{Float64}}()

  # Set user data
  userdata = userdata_type(4.0, 10.0, 1000.0)

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

    # Initialize DGO
    dgo_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing
    @reset control[].maxit = 2500
    # @reset control[].trb_control[].maxit = 100
    # @reset control[].print_level = 1

    # Start from 0
    x = Float64[0, 0, 0]

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      dgo_import(control, data, status, n, x_l, x_u,
                 "coordinate", ne, H_row, H_col, Cint[])

      dgo_solve_with_mat(data, userdata, status, n, x, g,
                         ne, fun, grad, hess, hessprod, prec)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      dgo_import(control, data, status, n, x_l, x_u,
                 "sparse_by_rows", ne, Cint[], H_col, H_ptr)

      dgo_solve_with_mat(data, userdata, status, n, x, g,
                         ne, fun, grad, hess, hessprod, prec)
    end

    # dense
    if d == 3
      st = 'D'
      dgo_import(control, data, status, n, x_l, x_u,
                 "dense", ne, Cint[], Cint[], Cint[])

      dgo_solve_with_mat(data, userdata, status, n, x, g,
                         ne, fun, grad, hess_dense, hessprod, prec)
    end

    # diagonal
    if d == 4
      st = 'I'
      dgo_import(control, data, status, n, x_l, x_u,
                 "diagonal", ne, Cint[], Cint[], Cint[])

      dgo_solve_with_mat(data, userdata, status, n, x, g,
                         ne, fun_diag, grad_diag, hess_diag,
                         hessprod_diag, prec)
    end

    # access by products
    if d == 5
      st = 'P'
      dgo_import(control, data, status, n, x_l, x_u,
                 "absent", ne, Cint[], Cint[], Cint[])

      dgo_solve_without_mat(data, userdata, status, n, x, g,
                            fun, grad, hessprod, shessprod, prec)
    end

    # Record solution information
    dgo_information(data, inform, status)

    if inform[].status[] == 0
      @printf("%c:%6i evaluations. Optimal objective value = %5.2f status = %1i\n", st,
              inform[].f_eval, inform[].obj, inform[].status)
    elseif inform[].status[] == -18
      @printf("%c:%6i evaluations. Best objective value = %5.2f status = %1i\n", st,
              inform[].f_eval, inform[].obj, inform[].status)
    else
      @printf("%c: DGO_solve exit status = %1i\n", st, inform[].status)
    end

    # @printf("x: ")
    # for i in 1:n
    #   @printf("%f ", x[i])
    # end
    # @printf("\n")
    # @printf("gradient: ")
    # for i in 1:n
    #   @printf("%f ", g[i])
    # end
    # @printf("\n")

    # Delete internal workspace
    dgo_terminate(data, control, inform)
  end

  @printf("\n tests reverse-communication options\n\n")

  # reverse-communication input/output
  eval_status = Ref{Cint}()
  nnz_u = Ref{Cint}()
  nnz_v = Ref{Cint}()
  f = 0.0
  u = zeros(Float64, n)
  v = zeros(Float64, n)
  index_nz_u = zeros(Cint, n)
  index_nz_v = zeros(Cint, n)
  # real_wp_ H_val[ne], H_dense[n*(n+1)/2], H_diag[n]

  for d in 1:5
    # Initialize DGO
    dgo_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing
    @reset control[].maxit = 2500
    # @reset control[].trb_control[].maxit = 100
    # @reset control[].print_level = 1

    # Start from 0
    x = Float64[0.0, 0.0, 0.0]

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      dgo_import(control, data, status, n, x_l, x_u, "coordinate", ne, H_row, H_col, Cint[])

      terminated = false
      while !terminated # reverse-communication loop
        dgo_solve_reverse_with_mat(data, status, eval_status, n, x, f, g, ne, H_val, u, v)
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
        elseif status[] == 5 # evaluate Hv product
          eval_status = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status = prec(n, x, u, v, userdata)
        elseif status[] == 23 # evaluate f and g
          eval_status = fun(n, x, f, userdata)
          eval_status = grad(n, x, g, userdata)
        elseif status[] == 25 # evaluate f and Hv product
          eval_status = fun(n, x, f, userdata)
          eval_status = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 35 # evaluate g and Hv product
          eval_status = grad(n, x, g, userdata)
          eval_status = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 235 # evaluate f, g and Hv product
          eval_status = fun(n, x, f, userdata)
          eval_status = grad(n, x, g, userdata)
          eval_status = hessprod(n, x, u, v, false, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # sparse by rows
    if d == 2
      st = 'R'
      dgo_import(control, data, status, n, x_l, x_u,
                 "sparse_by_rows", ne, Cint[], H_col, H_ptr)

      terminated = false
      while !terminated # reverse-communication loop
        dgo_solve_reverse_with_mat(data, status, eval_status,
                                   n, x, f, g, ne, H_val, u, v)
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
        elseif status[] == 5 # evaluate Hv product
          eval_status = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status = prec(n, x, u, v, userdata)
        elseif status[] == 23 # evaluate f and g
          eval_status = fun(n, x, f, userdata)
          eval_status = grad(n, x, g, userdata)
        elseif status[] == 25 # evaluate f and Hv product
          eval_status = fun(n, x, f, userdata)
          eval_status = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 35 # evaluate g and Hv product
          eval_status = grad(n, x, g, userdata)
          eval_status = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 235 # evaluate f, g and Hv product
          eval_status = fun(n, x, f, userdata)
          eval_status = grad(n, x, g, userdata)
          eval_status = hessprod(n, x, u, v, false, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # dense
    if d == 3
      st = 'D'
      dgo_import(control, data, status, n, x_l, x_u,
                 "dense", ne, Cint[], Cint[], Cint[])

      terminated = false
      while !terminated # reverse-communication loop
        dgo_solve_reverse_with_mat(data, status, eval_status,
                                   n, x, f, g, n * (n + 1) / 2,
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
          eval_status = hess_dense(n, n * (n + 1) / 2, x, H_dense,
                                   userdata)
        elseif status[] == 5 # evaluate Hv product
          eval_status = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status = prec(n, x, u, v, userdata)
        elseif status[] == 23 # evaluate f and g
          eval_status = fun(n, x, f, userdata)
          eval_status = grad(n, x, g, userdata)
        elseif status[] == 25 # evaluate f and Hv product
          eval_status = fun(n, x, f, userdata)
          eval_status = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 35 # evaluate g and Hv product
          eval_status = grad(n, x, g, userdata)
          eval_status = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 235 # evaluate f, g and Hv product
          eval_status = fun(n, x, f, userdata)
          eval_status = grad(n, x, g, userdata)
          eval_status = hessprod(n, x, u, v, false, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # diagonal
    if d == 4
      st = 'I'
      dgo_import(control, data, status, n, x_l, x_u,
                 "diagonal", ne, Cint[], Cint[], Cint[])

      terminated = false
      while !terminated # reverse-communication loop
        dgo_solve_reverse_with_mat(data, status, eval_status,
                                   n, x, f, g, n, H_diag, u, v)
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
        elseif status[] == 5 # evaluate Hv product
          eval_status = hessprod_diag(n, x, u, v, false,
                                      userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status = prec(n, x, u, v, userdata)
        elseif status[] == 23 # evaluate f and g
          eval_status = fun_diag(n, x, f, userdata)
          eval_status = grad_diag(n, x, g, userdata)
        elseif status[] == 25 # evaluate f and Hv product
          eval_status = fun_diag(n, x, f, userdata)
          eval_status = hessprod_diag(n, x, u, v, false,
                                      userdata)
        elseif status[] == 35 # evaluate g and Hv product
          eval_status = grad_diag(n, x, g, userdata)
          eval_status = hessprod_diag(n, x, u, v, false,
                                      userdata)
        elseif status[] == 235 # evaluate f, g and Hv product
          eval_status = fun_diag(n, x, f, userdata)
          eval_status = grad_diag(n, x, g, userdata)
          eval_status = hessprod_diag(n, x, u, v, false,
                                      userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # access by products
    if d == 5
      st = 'P'
      dgo_import(control, data, status, n, x_l, x_u,
                 "absent", ne, Cint[], Cint[], Cint[])

      nnz_u = 0
      terminated = false
      while !terminated # reverse-communication loop
        dgo_solve_reverse_without_mat(data, status, eval_status,
                                      n, x, f, g, u, v, index_nz_v,
                                      nnz_v, index_nz_u, nnz_u)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # evaluate f
          eval_status = fun(n, x, f, userdata)
        elseif status[] == 3 # evaluate g
          eval_status = grad(n, x, g, userdata)
        elseif status[] == 5 # evaluate Hv product
          eval_status = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 6 # evaluate the product with P
          eval_status = prec(n, x, u, v, userdata)
        elseif status[] == 7 # evaluate sparse Hess-vect product
          eval_status = shessprod(n, x, nnz_v, index_nz_v, v,
                                  nnz_u, index_nz_u, u,
                                  false, userdata)
        elseif status[] == 23 # evaluate f and g
          eval_status = fun(n, x, f, userdata)
          eval_status = grad(n, x, g, userdata)
        elseif status[] == 25 # evaluate f and Hv product
          eval_status = fun(n, x, f, userdata)
          eval_status = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 35 # evaluate g and Hv product
          eval_status = grad(n, x, g, userdata)
          eval_status = hessprod(n, x, u, v, false, userdata)
        elseif status[] == 235 # evaluate f, g and Hv product
          eval_status = fun(n, x, f, userdata)
          eval_status = grad(n, x, g, userdata)
          eval_status = hessprod(n, x, u, v, false, userdata)
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end
    end

    # Record solution information
    dgo_information(data, inform, status)

    if inform[].status[] == 0
      @printf("%c:%6i evaluations. Optimal objective value = %5.2f status = %1i\n", st,
              inform[].f_eval, inform[].obj, inform[].status)
    elseif inform[].status[] == -18
      @printf("%c:%6i evaluations. Best objective value = %5.2f status = %1i\n", st,
              inform[].f_eval, inform[].obj, inform[].status)
    else
      @printf("%c: DGO_solve exit status = %1i\n", st, inform[].status)
    end

    # @printf("x: ")
    # for i in 1:n
    #   @printf("%f ", x[i])
    # end
    # @printf("\n")
    # @printf("gradient: ")
    # for i in 1:n
    #   @printf("%f ", g[i])
    # end
    # @printf("\n")

    # Delete internal workspace
    dgo_terminate(data, control, inform)
  end
end

@testset "DGO" begin
  @test test_dgo() == 0
end
