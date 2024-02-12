# test_blls.jl
# Simple code to test the Julia interface to BLLS

using GALAHAD
using Test
using Printf
using Accessors

# Custom userdata struct
struct userdata_type
  scale::Float64
end

function test_blls()
  # Apply preconditioner
  function prec(n::Int, var::Vector{Float64}, p::Vector{Float64}, userdata::userdata_type)
    scale = userdata.scale
    for i in 1:n
      p[i] = scale * v[i]
    end
    return 0
  end

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{blls_control_type{Float64}}()
  inform = Ref{blls_inform_type{Float64}}()

  # Set user data
  userdata = userdata_type(1.0)

  # Set problem data
  n = 10 # dimension
  o = n + 1 # number of residuals
  Ao_ne = 2 * n # sparse Jacobian elements
  Ao_dense_ne = o * n # dense Jacobian elements
  # row-wise storage
  Ao_row = zeros(Cint, Ao_ne) # row indices,
  Ao_col = zeros(Cint, Ao_ne) # column indices
  Ao_ptr_ne = o + 1 # number of row pointers
  Ao_ptr = zeros(Cint, Ao_ptr_ne)  # row pointers
  Ao_val = zeros(Float64, Ao_ne) # values
  Ao_dense = zeros(Float64, Ao_dense_ne) # dense values
  # column-wise storage
  Ao_by_col_row = zeros(Cint, Ao_ne) # row indices,
  Ao_by_col_ptr_ne = n + 1 # number of column pointers
  Ao_by_col_ptr = zeros(Cint, Ao_by_col_ptr_ne)  # column pointers
  Ao_by_col_val = zeros(Float64, Ao_ne) # values
  Ao_by_col_dense = zeros(Float64, Ao_dense_ne) # dense values
  b = zeros(Float64, o)  # linear term in the objective
  x_l = zeros(Float64, n) # variable lower bound
  x_u = zeros(Float64, n) # variable upper bound
  x = zeros(Float64, n) # variables
  z = zeros(Float64, n) # dual variables
  r = zeros(Float64, o) # residual
  g = zeros(Float64, n) # gradient
  w = zeros(Float64, o) # weights

  # Set output storage
  x_stat = zeros(Cint, n) # variable status
  st = ' '
  status = Ref{Cint}()

  x_l[1] = -1.0
  for i in 2:n
    x_l[i] = -Inf
  end
  x_u[1] = 1.0
  x_u[2] = Inf
  for i in 3:n
    x_u[i] = 2.0
  end

  #   A = ( I )  and b = (i * e)
  #       (e^T)          (n + 1)

  for i in 1:n
    b[i] = i + 1
  end
  b[n + 1] = n + 1

  w[1] = 2.0
  w[2] = 1.0
  for i in 3:o
    w[i] = 1.0
  end

  # # A by rows
  for i in 1:n
    Ao_ptr[i] = i + 1
    Ao_row[i] = i + 1
    Ao_col[i] = i + 1
    Ao_val[i] = 1.0
  end
  Ao_ptr[n + 1] = n + 1
  for i in 1:n
    Ao_row[n + i] = o
    Ao_col[n + i] = i + 1
    Ao_val[n + i] = 1.0
  end
  Ao_ptr[o] = Ao_ne + 1
  l = 0
  for i in 1:n
    for j in 1:n
      l = l + 1
      if i == j
        Ao_dense[l] = 1.0
      else
        Ao_dense[l] = 0.0
      end
    end
  end
  for j in 1:n
    l = l + 1
    Ao_dense[l] = 1.0
  end

  # # A by columns
  l = 0
  for j in 1:n
    l = l + 1
    Ao_by_col_ptr[j] = l + 1
    Ao_by_col_row[l] = j + 1
    Ao_by_col_val[l] = 1.0
    l = l + 1
    Ao_by_col_row[l] = o
    Ao_by_col_val[l] = 1.0
  end
  Ao_by_col_ptr[n + 1] = Ao_ne + 1
  l = 0
  for j in 1:n
    for i in 1:n
      l = l + 1
      if i == j
        Ao_by_col_dense[l] = 1.0
      else
        Ao_by_col_dense[l] = 0.0
      end
    end
    l = l + 1
    Ao_by_col_dense[l] = 1.0
  end

  @printf(" fortran sparse matrix indexing\n\n")
  @printf(" basic tests of blls storage formats\n\n")

  for d in 1:5

    # Initialize BLLS
    blls_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # fortran sparse matrix indexing

    # Start from 0
    for i in 1:n
      x[i] = 0.0
      z[i] = 0.0
    end

    # sparse co-ordinate storage
    if d == 1
      st = "CO"
      blls_import(control, data, status, n, o,
                  "coordinate", Ao_ne, Ao_row, Ao_col, 0, Cint[])

      blls_solve_given_a(data, userdata, status, n, o,
                         Ao_ne, Ao_val, b, x_l, x_u,
                         x, z, r, g, x_stat, w, prec)
    end

    # sparse by rows
    if d == 2
      st = "SR"
      blls_import(control, data, status, n, o,
                  "sparse_by_rows", Ao_ne, Cint[], Ao_col,
                  Ao_ptr_ne, Ao_ptr)

      blls_solve_given_a(data, userdata, status, n, o,
                         Ao_ne, Ao_val, b, x_l, x_u,
                         x, z, r, g, x_stat, w, prec)
    end

    # dense by rows
    if d == 3
      st = "DR"
      blls_import(control, data, status, n, o,
                  "dense_by_rows", Ao_dense_ne,
                  Cint[], Cint[], 0, Cint[])

      blls_solve_given_a(data, userdata, status, n, o,
                         Ao_dense_ne, Ao_dense, b, x_l, x_u,
                         x, z, r, g, x_stat, w, prec)
    end

    # sparse by columns
    if d == 4
      st = "SC"
      blls_import(control, data, status, n, o,
                  "sparse_by_columns", Ao_ne, Ao_by_col_row,
                  Cint[], Ao_by_col_ptr_ne, Ao_by_col_ptr)

      blls_solve_given_a(data, userdata, status, n, o,
                         Ao_ne, Ao_by_col_val, b, x_l, x_u,
                         x, z, r, g, x_stat, w, prec)
    end

    # dense by columns
    if d == 5
      st = "DC"
      blls_import(control, data, status, n, o,
                  "dense_by_columns", Ao_dense_ne,
                  Cint[], Cint[], 0, Cint[])

      blls_solve_given_a(data, userdata, status, n, o,
                         Ao_dense_ne, Ao_by_col_dense, b, x_l, x_u,
                         x, z, r, g, x_stat, w, prec)
    end

    blls_information(data, inform, status)

    if inform[].status == 0
      @printf("%s:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
              st, inform[].iter, inform[].obj, inform[].status)
    else
      @printf("%s: BLLS_solve exit status = %1i\n", st, inform[].status)
    end

    # @printf("x: ")
    # for i = 1:n
    #   @printf("%f ", x[i])
    # @printf("\n")
    # @printf("gradient: ")
    # for i = 1:n
    #   @printf("%f ", g[i])
    # @printf("\n")

    # Delete internal workspace
    blls_terminate(data, control, inform)
    @printf("\n tests reverse-communication options\n\n")
  end

  # reverse-communication input/output
  on = max(o, n)
  # int eval_status, nz_v_start, nz_v_end, nz_p_end
  # int nz_v[on], nz_p[o], mask[o]
  # real_wp_ v[on], p[on]

  nz_p_end = 0

  # Initialize BLLS
  blls_initialize(data, control, status)

  # Set user-defined control options
  @reset control[].f_indexing = true # fortran sparse matrix indexing

  # Start from 0
  for i in 1:n
    x[i] = 0.0
    z[i] = 0.0
  end

  st = "RC"
  for i in 1:o
    mask[i] = 0
  end
  blls_import_without_a(control, data, status, n, o)

  terminated = false
  while !terminated # reverse-communication loop
    blls_solve_reverse_a_prod(data, status, eval_status, n, o, b,
                              x_l, x_u, x, z, r, g, x_stat, v, p,
                              nz_v, nz_v_start, nz_v_end,
                              nz_p, nz_p_end, w)

    if status[] == 0 # successful termination
      terminated = true
    elseif status[] < 0 # error exit
      terminated = true
    elseif status[] == 2 # evaluate p = Av
      p[n + 1] = 0.0
      for i in 1:n
        p[i] = v[i]
        p[n + 1] = p[n + 1] + v[i]
      end
    elseif status[] == 3 # evaluate p = A^Tv
      for i in 1:n
        p[i] = v[i] + v[n + 1]
      end
    elseif status[] == 4 # evaluate p = Av for sparse v
      p[n + 1] = 0.0
      for i in 1:n
        p[i] = 0.0
      end
      for l in (nz_v_start - 1):nz_v_end
        i = nz_v[l]
        p[i] = v[i]
        p[n + 1] = p[n + 1] + v[i]
      end
    elseif status[] == 5 # evaluate p = sparse Av for sparse v
      nz_p_end = 0
      for l in (nz_v_start - 1):nz_v_end
        i = nz_v[l] - 1
        if mask[i] == 0
          mask[i] = 1
          nz_p[nz_p_end] = i + 1
          nz_p_end = nz_p_end + 1
          p[i] = v[i]
        end
        if mask[n + 1] == 0
          mask[n + 1] = 1
          nz_p[nz_p_end] = o
          nz_p_end = nz_p_end + 1
          p[n + 1] = v[i]
        else
          p[n + 1] = p[n + 1] + v[i]
        end
        for l in 1:nz_p_end
          mask[nz_p[l] - 1] = 0
        end
      end
    elseif status[] == 6 # evaluate p = sparse A^Tv
      for l in (nz_v_start - 1):nz_v_end
        i = nz_v[l] - 1
        p[i] = v[i] + v[n + 1]
      end
    elseif status[] == 7 # evaluate p = P^{-}v
      for i in 1:n
        p[i] = userdata.scale * v[i]
      end
    else
      @printf(" the value %1i of status should not occur\n", status)
    end
    eval_status = 0
  end

  # Record solution information
  blls_information(data, inform, status)

  # Print solution details
  if inform[].status == 0
    @printf("%s:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
            st, inform[].iter, inform[].obj, inform[].status)
  else
    @printf("%s: BLLS_solve exit status = %1i\n", st, inform[].status)
  end

  # @printf("x: ")
  # for i = 1:n
  #   @printf("%f ", x[i])
  # @printf("\n")
  # @printf("gradient: ")
  # for i = 1:n
  #   @printf("%f ", g[i])
  # @printf("\n")

  # Delete internal workspace
  return blls_terminate(data, control, inform)
end

@testset "BLLS" begin
  @test test_blls() == 0
end
