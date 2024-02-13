# test_bqp.jl
# Simple code to test the Julia interface to BQP

using GALAHAD
using Test
using Printf
using Accessors

function test_bqp()
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{bqp_control_type{Float64}}()
  inform = Ref{bqp_inform_type{Float64}}()

  # Set problem data
  n = 10 # dimension
  H_ne = 2 * n - 1 # Hesssian elements, NB lower triangle
  H_dense_ne = div(n * (n + 1), 2) # dense Hessian elements
  H_row = zeros(Cint, H_ne) # row indices,
  H_col = zeros(Cint, H_ne) # column indices
  H_ptr = zeros(Cint, n + 1)  # row pointers
  H_val = zeros(Float64, H_ne) # values
  H_dense = zeros(Float64, H_dense_ne) # dense values
  H_diag = zeros(Float64, n)   # diagonal values
  g = zeros(Float64, n)  # linear term in the objective
  f = 1.0  # constant term in the objective
  x_l = zeros(Float64, n) # variable lower bound
  x_u = zeros(Float64, n) # variable upper bound
  x = zeros(Float64, n) # variables
  z = zeros(Float64, n) # dual variables

  # Set output storage
  x_stat = zeros(Cint, n) # variable status
  st = ' '
  status = Ref{Cint}()

  g[1] = 2.0
  for i in 2:n
    g[i] = 0.0
  end
  x_l[1] = -1.0
  for i in 2:n
    x_l[i] = -Inf
  end
  x_u[1] = 1.0
  x_u[2] = Inf
  for i in 3:n
    x_u[i] = 2.0
  end

  # H = tridiag(2,1), H_dense = diag(2)
  l = 1
  H_ptr[1] = l
  H_row[l] = 1
  H_col[l] = 1
  H_val[l] = 2.0
  for i in 2:n
    l = l + 1
    H_ptr[i] = l
    H_row[l] = i
    H_col[l] = i - 1
    H_val[l] = 1.0
    l = l + 1
    H_row[l] = i
    H_col[l] = i
    H_val[l] = 2.0
  end
  H_ptr[n + 1] = l + 1

  l = 0
  for i in 1:n
    H_diag[i] = 2.0
    for j in 1:i
      l = l + 1
      if j < i - 1
        H_dense[l] = 0.0
      elseif j == i - 1
        H_dense[l] = 1.0
      else
        H_dense[l] = 2.0
      end
    end
  end

  @printf(" fortran sparse matrix indexing\n\n")
  @printf(" basic tests of bqp storage formats\n\n")

  for d in 1:4

    # Initialize BQP
    bqp_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # fortran sparse matrix indexing

    # Start from 0
    for i in 1:n
      x[i] = 0.0
      z[i] = 0.0
    end

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      bqp_import(control, data, status, n,
                 "coordinate", H_ne, H_row, H_col, C_NULL)

      bqp_solve_given_h(data, status, n, H_ne, H_val, g, f,
                        x_l, x_u, x, z, x_stat)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      bqp_import(control, data, status, n,
                 "sparse_by_rows", H_ne, C_NULL, H_col, H_ptr)

      bqp_solve_given_h(data, status, n, H_ne, H_val, g, f,
                        x_l, x_u, x, z, x_stat)
    end

    # dense
    if d == 3
      st = 'D'
      bqp_import(control, data, status, n,
                 "dense", H_dense_ne, C_NULL, C_NULL, C_NULL)

      bqp_solve_given_h(data, status, n, H_dense_ne, H_dense,
                        g, f, x_l, x_u, x, z, x_stat)
    end

    # diagonal
    if d == 4
      st = 'L'
      bqp_import(control, data, status, n,
                 "diagonal", H_ne, C_NULL, C_NULL, C_NULL)

      bqp_solve_given_h(data, status, n, n, H_diag, g, f,
                        x_l, x_u, x, z, x_stat)
    end

    bqp_information(data, inform, status)

    if inform[].status == 0
      @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n", st,
              inform[].iter, inform[].obj, inform[].status)
    else
      @printf("%c: BQP_solve exit status = %1i\n", st, inform[].status)
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
    bqp_terminate(data, control, inform)
  end

  @printf("\n tests reverse-communication options\n\n")

  # reverse-communication input/output
  nz_v_start = Ref{Cint}()
  nz_v_end = Ref{Cint}()
  nz_v = zeros(Cint, n)
  nz_prod = zeros(Cint, n)
  mask = zeros(Cint, n)
  v = zeros(Float64, n)
  prod = zeros(Float64, n)

  nz_prod_end = 1

  # Initialize BQP
  bqp_initialize(data, control, status)
  # @reset control[].print_level = 1

  # Set user-defined control options
  @reset control[].f_indexing = true # fortran sparse matrix indexing

  # Start from 0
  for i in 1:n
    x[i] = 0.0
    z[i] = 0.0
  end

  st = 'I'
  for i in 1:n
    mask[i] = 0
  end
  bqp_import_without_h(control, data, status, n)

  terminated = false
  while !terminated # reverse-communication loop
    bqp_solve_reverse_h_prod(data, status, n, g, f, x_l, x_u, x, z, x_stat, v, prod, nz_v,
                             nz_v_start, nz_v_end, nz_prod, nz_prod_end)

    if status[] == 0 # successful termination
      terminated = true
    elseif status[] < 0 # error exit
      terminated = true
    elseif status[] == 2 # evaluate Hv
      prod[1] = 2.0 * v[1] + v[2]
      for i in 2:n-1
        prod[i] = 2.0 * v[i] + v[i - 1] + v[i + 1]
      end
      prod[n] = 2.0 * v[n] + v[n - 1]
    elseif status[] == 3 # evaluate Hv for sparse v
      for i in 1:n
        prod[i] = 0.0
      end
      for l in nz_v_start[]:nz_v_end[]
        i = nz_v[l]
        if i > 1
          prod[i - 1] = prod[i - 1] + v[i]
        end
        prod[i] = prod[i] + 2.0 * v[i]
        if i < n
          prod[i + 1] = prod[i + 1] + v[i]
        end
      end
    elseif status[] == 4 # evaluate sarse Hv for sparse v
      nz_prod_end = 1
      for l in nz_v_start[]:nz_v_end[]
        i = nz_v[l]
        if i > 1
          if mask[i - 1] == 0
            mask[i - 1] = 1
            nz_prod[nz_prod_end] = i - 1
            nz_prod_end = nz_prod_end + 1
            prod[i - 1] = v[i]
          else
            prod[i - 1] = prod[i - 1] + v[i]
          end
        end
        if mask[i] == 0
          mask[i] = 1
          nz_prod[nz_prod_end] = i
          nz_prod_end = nz_prod_end + 1
          prod[i] = 2.0 * v[i]
        else
          prod[i] = prod[i] + 2.0 * v[i]
        end
        if i < n
          if mask[i + 1] == 0
            mask[i + 1] = 1
            nz_prod[nz_prod_end] = i + 1
            nz_prod_end = nz_prod_end + 1
          end
          prod[i + 1] = prod[i + 1] + v[i]
        end
        for l in 1:nz_prod_end
          mask[nz_prod[l]] = 0
        end
      end
    else
      @printf(" the value %1i of status should not occur\n", status[])
    end
  end

  # Record solution information
  bqp_information(data, inform, status)

  # Print solution details
  if inform[].status == 0
    @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n", st,
            inform[].iter, inform[].obj, inform[].status)
  else
    @printf("%c: BQP_solve exit status = %1i\n", st, inform[].status)
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
  bqp_terminate(data, control, inform)
  return 0
end

@testset "BQP" begin
  @test test_bqp() == 0
end
